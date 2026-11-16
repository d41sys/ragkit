#!/bin/bash
# Issue #11 (docs/agent_eval_gd_protocol.md): G main + redo, then D (a01 first) main + redo, then judge
# G and D live (A and E replayed from the cache). One stream. GPU out-of-memory passes are discarded (exit 3).
R=/Users/d41sy/Desktop/tue/coding/ai-engineering-capstones/ragkit
H=/Users/d41sy/Desktop/tue/coding/wt/ezcross-eu-law-e/harness/packages/ezc-cli  # feat/eu-law-answerable @ 2ae4af62, E's driver
A=$R/runs/agent_eval
UV="uv run --offline --extra st"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
oom() { grep -q "out of memory" "$1" && { mv "$1" "${1%.jsonl}.oom-$(date -u +%Y%m%dT%H%M%SZ).jsonl"; echo "OOM rows in $1: pass discarded"; exit 3; }; }
pgrep -f "21434:localhost:11434" >/dev/null || ssh -o BatchMode=yes -o ServerAliveInterval=30 -f -N -L 21434:localhost:11434 Korea
cd $R
$UV python -m ragkit.agentbench serve --setup G --port 8106 > $A/G.server.log 2>&1 &
until curl -s -o /dev/null http://127.0.0.1:8106/; do sleep 5; done
cd $H
for pass in main redo; do
  extra=""; [ $pass = redo ] && extra="--redo-failed"
  EZC_LOCAL_BASE_URL=http://localhost:21434/v1 bun src/eval/eu-law-run.ts --questions $A/questions.jsonl --num-ctx 65536 $extra \
    --backend http://127.0.0.1:8106/api/v1 --model "local/qwen3.6:27b-ctx64k" --out $A/G.jsonl > $A/G_$pass.log 2>&1
  echo "G $pass done $(date -u +%FT%TZ)"; oom $A/G.jsonl
done
pkill -f "agentbench serve --setup G"
cd $R
$UV python -m ragkit.agentbench ask-full --model qwen3.6:27b-ctx128k --only a01 > $A/D_a01.log 2>&1
echo "D a01: $(tail -1 $A/D_a01.log)"; oom $A/D.jsonl
if grep -q '"error"' $A/D.jsonl; then echo "D does not run on a01: recorded as not run"; else
  $UV python -m ragkit.agentbench ask-full --model qwen3.6:27b-ctx128k > $A/D_main.log 2>&1; echo "D main done $(date -u +%FT%TZ)"; oom $A/D.jsonl
  $UV python -m ragkit.agentbench ask-full --model qwen3.6:27b-ctx128k --redo-failed > $A/D_redo.log 2>&1; echo "D redo done $(date -u +%FT%TZ)"; oom $A/D.jsonl
fi
S=A,E,G; grep -q '"error"' $A/D.jsonl 2>/dev/null && [ "$(wc -l < $A/D.jsonl)" -le 1 ] || S=A,E,G,D
$UV python -m ragkit.agentbench report --live --setups $S --workers 1 --out docs/sota-runs/agent_eval_gd.txt --verdicts-out docs/sota-runs/agent_eval_gd_verdicts.json
echo "report done $(date -u +%FT%TZ)"
