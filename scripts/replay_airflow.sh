#!/usr/bin/env bash
# Replay every weekly run (May 2005 to Sep 2006) as a real Airflow DAG run, in order.
# Needs: AIRFLOW_HOME, AIRFLOW__CORE__DAGS_FOLDER, PVP_PYTHON, PVP_FEATURES, PVP_STATE set.
set -uo pipefail
AIRFLOW="${AIRFLOW_BIN:-airflow}"
LOG_DIR="${PVP_STATE}/airflow_runs"
mkdir -p "$LOG_DIR"
dates=$("$PVP_PYTHON" -c "from pvprod import steps, timeline; print(' '.join(str(d) for d in timeline.run_dates(steps.load_features())))")
start=$(date +%s); ok=0; failed=0
for d in $dates; do
  if "$AIRFLOW" dags test player_value_weekly "$d" > "$LOG_DIR/$d.log" 2>&1 \
     && [ "$(grep -c 'new_state=success' "$LOG_DIR/$d.log")" -eq 6 ]; then
    ok=$((ok + 1)); echo "$d ok"
  else
    failed=$((failed + 1)); echo "$d FAILED"
  fi
done
echo "runs_ok=$ok runs_failed=$failed seconds=$(( $(date +%s) - start ))"
