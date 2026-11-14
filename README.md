# Player Value in Production: weekly scoring, drift monitoring and gated retraining

[player-value-model](https://github.com/GurramVardhanReddy/player-value-model) showed that a
gradient-boosting model can pick out, from a player's first 90 days, the players who will be worth
the most over the next year. This project runs that model the way a sportsbook would: as a weekly
Airflow pipeline with an MLflow model registry, drift monitoring, monthly retraining that has to
beat the live model before it replaces it, a scoring API on Google Cloud Run, and scores
published to Snowflake for CRM.

It is replayed over real history: daily betting records of bwin customers (bwin is now part of
Entain) from the Transparency Project (Division on Addiction), played back week by week from May
2005 to September 2006, so every decision the pipeline makes uses only what it could have known on
that date.

## What happens each Monday

```
ensure_champion -> score_new_batches -> evaluate_matured -> maybe_retrain -> export_champion -> publish_scores
                          \______________________________________________________________________/
```

1. **Score.** Players whose first 90 days just ended are scored by the current champion (the MLflow
   model version with the `champion` alias): expected next-year turnover, and GGR = turnover × house
   margin.
2. **Monitor drift.** Each new batch is compared with the champion's training players: PSI per
   feature (10 quantile bins) and on the score itself; warn at 0.10, alert at 0.25.
3. **Grade.** A batch's real outcome arrives a year after it was scored. It is graded with the model
   version that actually scored it: share of next-year GGR captured by the top 10%, turnover Gini,
   and total forecast vs actual.
4. **Retrain (first Monday of the month).** A challenger is trained on all outcomes known so far
   except the newest 4 weeks, which are held out. It is promoted only if it captures at least as
   much GGR in its top 10% on that holdout **and** its total turnover forecast is no further from
   actual than the champion's.
5. **Export and publish.** The champion is exported to the scoring service, and new scores are
   written to `MARKETING.PLAYER_VALUE.SCORES` in Snowflake by a least-privilege role.

Tasks run in the project's own Python environment (`@task.external_python`), so Airflow's
dependencies never clash with MLflow and scikit-learn.

## Results of the replay

All numbers come from `results/replay_summary.json` and `results/batches.csv`.

- **68 weekly Airflow runs**, all succeeded (`airflow dags test`, one run per Monday, in order).
- **29,472 players scored** in 16 weekly batches. The first champion was trained on the first two
  weeks of players (11,775) as a stand-in prior cohort.
- **The label delay decides everything.** Because outcomes take a year, every production batch
  was scored by v1 before a single outcome arrived. For that whole year drift monitoring is the only
  early warning there is.
- **Drift alerts fired on 14 of 16 batches**, every time on `reg_to_first_bet` and
  `reg_to_first_dep` (days from registration to first bet / first deposit). This is real: the
  dataset is a cohort that registered in February 2005, so players who first bet months later have
  long registration gaps the training players never had. The score distribution itself never
  reached the alert level (highest PSI 0.248, on a 94-player batch).
- **Accuracy held up anyway.** The top 10% of players picked captured 63.8% of next-year GGR on the
  first 4 batches and 62.1% on the last 4. Across batches, the worst feature's PSI and the
  top-10% capture were uncorrelated (Spearman 0.03). Drift on two timing features did not mean the
  model got worse, which is why retraining is gated on graded outcomes rather than on drift alone.
- **Gated retraining:** 3 challengers trained, 1 promoted.

| Challenger | Top-10% GGR share (challenger vs champion) | Total turnover forecast / actual | Promoted |
|---|---|---|---|
| v2 vs v1 | 70.0% vs 69.0% | 0.952 vs 1.019 | No: better ranking, worse total |
| v3 vs v1 | 62.8% vs 61.4% | 1.001 vs 1.062 | **Yes** |
| v4 vs v3 | 61.1% vs 60.3% | 0.687 vs 0.738 | No: better ranking, worse total |

  v3 is the final champion and the model behind the API.

**Limits.** Later batches are small (as few as 59 players), so their per-batch metrics are noisy;
holdouts shrink the same way (463 players for v4). The replay compresses one registration cohort
into weekly arrivals, so it shows how the pipeline behaves, not how a live sportsbook's traffic
looks.

## Scoring API (Google Cloud Run)

FastAPI service with the exported champion baked into the image, deployed to Cloud Run
(europe-west1, scales to zero): https://player-value-api-792548722064.europe-west1.run.app
(the first request after an idle spell takes a few seconds while an instance starts).

- `GET /health`
- `GET /model`: version, house margin and value-band edges of the champion
- `POST /score`: a list of first-90-day player summaries in, expected turnover, expected GGR and
  a value band out (1 = top 10% of the champion's training players)

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

The bwin data is not redistributed here; it is available from the Transparency Project. Build
the feature table with player-value-model, then:

```bash
pip install -e ".[pipeline,serve,dev]"
export PVP_FEATURES=/path/to/features.parquet PVP_STATE=./state
python -m pvprod.cli bootstrap
scripts/replay_airflow.sh        # every weekly run through Airflow (needs Airflow 3.1 in its own venv)
python -m pvprod.cli export
python -m pvprod.cli summary
gcloud run deploy player-value-api --source . --region europe-west1
```

Snowflake: `infra/main.tf` (Terraform) adds the `PLAYER_VALUE` schema to the `MARKETING` database
from [marketing-data-platform](https://github.com/GurramVardhanReddy/marketing-data-platform) and
lets its `TRANSFORMER` role create tables there. `publish_scores` runs only when
`SNOWFLAKE_ACCOUNT` is set; key-pair auth, no passwords.

## Tests and CI

GitHub Actions runs ruff and pytest (a synthetic end-to-end run of bootstrap, scoring, grading,
retraining and export, plus API tests), installs Airflow 3.1.3 and checks the DAG's task graph,
and builds the Docker image and calls `/health` and `/model` on it.
