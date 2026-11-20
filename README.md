# Production ML Pipeline: weekly scoring, drift monitoring and gated retraining

[customer-lifetime-value-model](https://github.com/GurramVardhanReddy/customer-lifetime-value-model)
showed that a gradient-boosting model can pick out, from a customer's first 90 days, the customers who
will be worth the most over the next year. This project runs that model in production as a weekly
**Apache Airflow** pipeline, with:
- an **MLflow model registry**;
- **drift monitoring** (PSI);
- monthly retraining whose challenger has to beat the live model before it replaces it;
- a **FastAPI scoring API on Google Cloud Run**;
- scores published to **Snowflake** for CRM.

The pipeline is replayed over real history, played back week by week from May 2005 to September 2006.
Every decision it makes uses only what it could have known on that date. The history is real daily
transaction records of an online entertainment platform's customers (a public research dataset; see
[Data](#data)).

## What happens each Monday

```
ensure_champion -> score_new_batches -> evaluate_matured -> maybe_retrain -> export_champion -> publish_scores
                          \______________________________________________________________________/
```

1. **Score.** Customers whose first 90 days just ended are scored by the current champion (the
   MLflow model version with the `champion` alias). The score is expected next-year spend, and
   expected revenue = spend × margin.
2. **Monitor drift.** Each new batch is compared with the champion's training customers: PSI per
   feature (10 quantile bins) and on the score itself. It warns at 0.10 and alerts at 0.25.
3. **Grade.** A batch's real outcome arrives a year after it was scored. It is graded with the model
   version that actually scored it, on three measures:
   - the share of next-year revenue captured by the top 10%;
   - spend Gini;
   - total forecast vs actual.
4. **Retrain (first Monday of the month).** A challenger is trained on all outcomes known so far,
   except the newest 4 weeks, which are held out. It is promoted only if both hold on that holdout:
   - its top 10% captures at least as much revenue as the champion's;
   - its total spend forecast is no further from actual than the champion's.
5. **Export and publish.** The champion is exported to the scoring service. A least-privilege role
   writes new scores to `MARKETING.PLAYER_VALUE.SCORES` in Snowflake.

Tasks run in the project's own Python environment (`@task.external_python`), so Airflow's
dependencies never clash with MLflow and scikit-learn.

## Results of the replay

All numbers come from `results/replay_summary.json` and `results/batches.csv`.

- **68 weekly Airflow runs, all succeeded** (`airflow dags test`, one run per Monday, in order).
- **29,472 customers scored** in 16 weekly batches. The first champion was trained on the first two
  weeks of customers (11,775) as a stand-in prior cohort.
- **The label delay decides everything.** Outcomes take a year to arrive, so every production batch
  was scored by v1 before a single outcome came in. For that whole year, drift monitoring is the only
  early warning there is.
- **Drift alerts fired on 14 of 16 batches**, every time on `reg_to_first_bet` and
  `reg_to_first_dep` (days from registration to first purchase / first deposit). This is real
  drift. The dataset is a cohort that registered in February 2005, so customers who first purchased
  months later have long registration gaps the training customers never had. The score distribution
  itself never reached the alert level (highest PSI 0.248, on a 94-customer batch).
- **Accuracy held up anyway.** The top 10% of customers picked captured 63.8% of next-year revenue on
  the first 4 batches and 62.1% on the last 4. Across batches, the worst feature's PSI and the
  top-10% capture were uncorrelated (Spearman 0.03). Drift on two timing features did not mean the
  model got worse. That is why retraining is gated on graded outcomes rather than on drift alone.
- **Gated retraining:** 3 challengers trained, 1 promoted.

| Challenger | Top-10% revenue share (challenger vs champion) | Total spend forecast / actual | Promoted |
|---|---|---|---|
| v2 vs v1 | 70.0% vs 69.0% | 0.952 vs 1.019 | No: better ranking, worse total |
| v3 vs v1 | 62.8% vs 61.4% | 1.001 vs 1.062 | **Yes** |
| v4 vs v3 | 61.1% vs 60.3% | 0.687 vs 0.738 | No: better ranking, worse total |

  v3 is the final champion and the model behind the API.

**Limits.**
- Later batches are small (as few as 59 customers), so their per-batch metrics are noisy.
- Holdouts shrink the same way (463 customers for v4).
- The replay compresses one registration cohort into weekly arrivals. It shows how the pipeline
  behaves, not how a live platform's traffic looks.

## Scoring API (Google Cloud Run)

A FastAPI service with the exported champion baked into the image, deployed to Cloud Run
(europe-west1, scales to zero): https://player-value-api-792548722064.europe-west1.run.app. The
first request after an idle spell takes a few seconds while an instance starts.

- `GET /health`
- `GET /model`: the champion's version, margin and value-band edges
- `POST /score`: takes a list of first-90-day customer summaries. It returns expected spend, expected
  revenue and a value band (1 = top 10% of the champion's training customers).

The request schema keeps the original dataset's field names (`player_id`, `stake_90`, and so on):

```bash
URL=https://player-value-api-792548722064.europe-west1.run.app
curl -s -X POST "$URL/score" -H 'content-type: application/json' -d '[{"player_id": 1,
  "first_bet_date": "2005-03-01", "stake_90": 900, "stake_live_90": 100, "win_90": 810,
  "bets_90": 60, "active_days_90": 25, "last_active_day": 85, "stake_wk1": 80, "stake_m1": 300,
  "stake_m3": 350, "active_days_m3": 9, "active_days_last14": 4, "max_day_stake": 90,
  "max_7d_stake": 160, "half_stake_day": 40, "live_days_90": 5, "country_id": 276, "gender": 1,
  "birth_year": 1975, "reg_to_first_bet": 3, "reg_to_first_dep": 1}]'
```

## Run it

Build the feature table with customer-lifetime-value-model, then:

```bash
pip install -e ".[pipeline,serve,dev]"
export PVP_FEATURES=/path/to/features.parquet PVP_STATE=./state
python -m pvprod.cli bootstrap
scripts/replay_airflow.sh        # every weekly run through Airflow (needs Airflow 3.1 in its own venv)
python -m pvprod.cli export
python -m pvprod.cli summary
gcloud run deploy player-value-api --source . --region europe-west1
```

**Snowflake.** `infra/main.tf` (Terraform) adds the `PLAYER_VALUE` schema to the `MARKETING`
database from [marketing-data-platform](https://github.com/GurramVardhanReddy/marketing-data-platform),
and lets its `TRANSFORMER` role create tables there. `publish_scores` runs only when
`SNOWFLAKE_ACCOUNT` is set; it uses key-pair auth, no passwords.

## Tests and CI

GitHub Actions runs three checks:
- ruff and pytest: a synthetic end-to-end run of bootstrap, scoring, grading, retraining and export,
  plus API tests;
- an install of Airflow 3.1.3 and a check of the DAG's task graph;
- a Docker image build that calls `/health` and `/model` on the image.

## Data

The history is the public Transparency Project dataset of the Division on Addiction (Cambridge Health
Alliance). It holds daily transaction records of customers of bwin, an online sports-betting operator
(now part of Entain). It is not redistributed here; it is available from the Transparency Project.
