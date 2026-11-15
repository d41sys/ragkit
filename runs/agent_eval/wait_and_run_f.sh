#!/bin/bash
# Waits until the shared Korea Ollama can load the agent model again (another user's jobs fill GPU2),
# then runs setup F (pipeline_f.sh). Polls every 5 min, gives up after 48 h. Log: wait_f.log
R=/Users/d41sy/Desktop/tue/coding/ai-engineering-capstones/ragkit/runs/agent_eval
cd /Users/d41sy/Desktop/tue/coding/ai-engineering-capstones/ragkit
end=$(( $(date +%s) + 48*3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  used=$(ssh -o BatchMode=yes Korea 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2' 2>/dev/null)
  echo "$(date -u +%FT%TZ) gpu2 used ${used:-?} MiB"
  if [ -n "$used" ] && [ "$used" -lt 23300 ]; then
    pgrep -f "21434:localhost:11434" >/dev/null || ssh -o BatchMode=yes -f -N -L 21434:localhost:11434 Korea
    if curl -s -m 600 http://localhost:21434/api/generate -d '{"model":"qwen3.6:27b-ctx64k","prompt":"hi","stream":false,"think":false,"options":{"num_predict":1}}' | grep -q '"done":true'; then
      echo "$(date -u +%FT%TZ) model loads; starting F"
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline --extra st python -m ragkit.agentbench serve --setup F --port 8105 > $R/F.server.log 2>&1 &
      until curl -s -o /dev/null http://127.0.0.1:8105/; do sleep 5; done
      bash $R/pipeline_f.sh; rc=$?
      pkill -f "agentbench serve --setup F"
      [ $rc -ne 3 ] && { pkill -f "21434:localhost:11434"; exit $rc; }
      echo "$(date -u +%FT%TZ) pass discarded after GPU OOM; waiting again"
      sleep 300; continue
    fi
    echo "$(date -u +%FT%TZ) load test failed"
  fi
  sleep 300
done
echo "gave up after 48 h"; exit 1
