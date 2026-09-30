#!/usr/bin/env bash
# Run every problem in sequence (3 workers, thermal guard 80/72 C), then build statistics, tables and figures.
# Resumable: completed runs are skipped. Stop at any time with:  kill -- -$(cat results/run_all.pid)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo $$ > results/run_all.pid
status=0
for problem in pendulum heat wave; do
    echo "$(date '+%F %T') === $problem ==="
    "$PY" scripts/run_experiment.py --problem "$problem" --workers 3 --pause-at 80 --resume-at 72 || status=1
done
echo "$(date '+%F %T') === tables and figures ==="
"$PY" scripts/make_tables_figs.py > results/make_tables_figs.log 2>&1 || status=1
echo "$(date '+%F %T') === finished (status=$status) ==="
rm -f results/run_all.pid
exit $status
