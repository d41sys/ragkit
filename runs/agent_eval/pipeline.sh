#!/bin/bash
# Detached finisher: wait for the running A/B drivers, re-run rows with an error once more (both
# setups, two streams), then judge A and B live and write the report. Log: runs/agent_eval/pipeline.log
R=/Users/d41sy/Desktop/tue/coding/ai-engineering-capstones/ragkit/runs/agent_eval
D=/Users/d41sy/Desktop/tue/coding/ezcross-ems-ragkit/harness/packages/ezc-cli
while pgrep -f "src/eval/eu-law-run.ts" > /dev/null; do sleep 30; done
echo "main passes done $(date -u +%FT%TZ)"
cd $D
for s in A:8101 B:8102; do n=${s%%:*}; p=${s##*:}
  EZC_LOCAL_BASE_URL=http://localhost:21434/v1 bun src/eval/eu-law-run.ts --questions $R/questions.jsonl --num-ctx 65536 --redo-failed --backend http://127.0.0.1:$p/api/v1 --model "local/qwen3.6:27b-ctx64k" --out $R/$n.jsonl > $R/redo_$n.log 2>&1 &
done
wait
echo "redo passes done $(date -u +%FT%TZ)"
cd /Users/d41sy/Desktop/tue/coding/ai-engineering-capstones/ragkit
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline --extra st python -m ragkit.agentbench report --live --setups A,B --workers 2 --out docs/sota-runs/agent_eval.txt --verdicts-out docs/sota-runs/agent_eval_verdicts.json
echo "report done $(date -u +%FT%TZ)"
