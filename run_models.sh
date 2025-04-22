#!/usr/bin/env bash
LOGFILE="vlmevalkit_$(date +%Y%m%dT%H%M%S).log"

for MODEL in \
    meta-llama/Llama-3.2-11B-Vision \
    meta-llama/Llama-3.2-11B-Vision-Instruct \
    meta-llama/Llama-3.2-90B-Vision \
    meta-llama/Llama-3.2-90B-Vision-Instruct \
	OpenGVLab/InternVL2-8B \
	google/gemma-3-27b-it \
	google/gemma-3-27b-it-qat-q4_0-gguf \
	stepfun-ai/GOT-OCR2_0 \
	openbmb/MiniCPM-Llama3-V-2_5
do
  echo
  echo "=== $MODEL @ $(date +'%Y-%m-%dT%H:%M:%S') ===" | tee -a "$LOGFILE"

  # 1) Launch LMDeploy server in the background
  lmdeploy serve api_server "$MODEL" --model-name "$MODEL" &
  SERVER_PID=$!

  # 2) Wait for it to be ready
  until curl -s http://localhost:8000/healthz >/dev/null; do
    sleep 1
  done

  # 3) Point VLMEvalKit at that server
  export LMDEPLOY_API_BASE="http://localhost:8000"

  # 4) Run your eval
  python run.py \
    --data gdt_vlmeval \
    --model lmdeploy \
    --remote-model-name "$MODEL" \
    --api-nproc 8 \
    --retry 3 \
    --verbose \
    --reuse \
  2>&1 | tee -a "$LOGFILE"

  # 5) Tear it down
  echo "Killing LMDeploy server (pid=$SERVER_PID)" | tee -a "$LOGFILE"
  kill $SERVER_PID

done