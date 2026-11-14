"""Publish scores to Snowflake (PLAYER_VALUE.SCORES) so CRM and marketing can use them.

Runs as the least-privilege TRANSFORMER role with key-pair auth. Re-publishing a batch replaces it.
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd

from pvprod import steps

TABLE = "MARKETING.PLAYER_VALUE.SCORES"


def _connect():
    import snowflake.connector

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
        authenticator="SNOWFLAKE_JWT", private_key_file=os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"],
        role=os.environ.get("SNOWFLAKE_ROLE", "TRANSFORMER"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "TRANSFORM_WH"))


def publish_scores(batches: list[str] | None = None) -> dict:
    """Upload scores for the given batches (default: all) and return row counts by batch."""
    if not os.environ.get("SNOWFLAKE_ACCOUNT"):
        return {"skipped": "SNOWFLAKE_ACCOUNT not set"}
    con = sqlite3.connect(steps.STATE / "monitor.db")
    df = pd.read_sql_query("select * from scores", con)
    con.close()
    if batches:
        df = df[df["batch"].isin(batches)]
    if df.empty:
        return {}
    sf = _connect()
    cur = sf.cursor()
    cur.execute(f"""create table if not exists {TABLE} (player_id number, batch date,
        scored_on date, model_version varchar, pred_value float, pred_turnover float,
        published_at timestamp_ntz default current_timestamp())""")
    for b, part in df.groupby("batch"):
        cur.execute(f"delete from {TABLE} where batch = %s", (b,))
        cur.executemany(
            f"insert into {TABLE} (player_id, batch, scored_on, model_version, pred_value, "
            "pred_turnover) values (%s, %s, %s, %s, %s, %s)",
            list(part[["player_id", "batch", "scored_on", "model_version", "pred_value",
                       "pred_turnover"]].itertuples(index=False, name=None)))
    counts = dict(cur.execute(f"select to_varchar(batch), count(*) from {TABLE} group by 1").fetchall())
    sf.close()
    return counts
