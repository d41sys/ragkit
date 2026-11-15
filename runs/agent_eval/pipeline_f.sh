#!/bin/bash
# Setup F: main pass, one redo pass of failed rows, then judge A and E (cached) and F live. One stream.
R=/Users/d41sy/Desktop/tue/coding/ai-engineering-capstones/ragkit/runs/agent_eval
D=/Users/d41sy/Desktop/tue/coding/ezcross-ems-ragkit/harness/packages/ezc-cli
cd $D
echo "main pass start $(date -u +%FT%TZ)"
for pass in main redo; do
  extra=""; [ $pass = redo ] && extra="--redo-failed"
  EZC_LOCAL_BASE_URL=http://localhost:21434/v1 bun src/eval/eu-law-run.ts --questions $R/questions.jsonl --num-ctx 65536 $extra --backend http://127.0.0.1:8105/api/v1 --model "local/qwen3.6:27b-ctx64k" --out $R/F.jsonl > $R/F_$pass.log 2>&1
  echo "$pass pass done $(date -u +%FT%TZ)"
  # GPU out of memory on the shared server is infrastructure, not model behaviour (protocol F note): discard the pass and wait again
  if grep -q "out of memory" $R/F.jsonl; then mv $R/F.jsonl $R/F.oom-$(date -u +%Y%m%dT%H%M%SZ).jsonl; echo "OOM rows: pass discarded"; exit 3; fi
done
cd /Users/d41sy/Desktop/tue/coding/ai-engineering-capstones/ragkit
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline --extra st python -m ragkit.agentbench report --live --setups A,E,F --workers 1 --out docs/sota-runs/agent_eval_f.txt --verdicts-out docs/sota-runs/agent_eval_f_verdicts.json
echo "report done $(date -u +%FT%TZ)"
