"""CLI: python -m pvprod.cli {bootstrap | replay | export | summary}

replay runs the same steps as the Airflow DAG in plain Python (no scheduler), for quick checks.
"""

import json
import sys

from pvprod import steps, timeline


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "bootstrap":
        print(steps.bootstrap())
    elif cmd == "replay":
        steps.bootstrap()
        for d in timeline.run_dates(steps.load_features()):
            print(d, json.dumps(steps.weekly(d), default=str)[:200])
    elif cmd == "export":
        print(json.dumps(steps.export_champion(), indent=2))
    elif cmd == "summary":
        from pvprod import report
        print(json.dumps(report.summary(), indent=2))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
