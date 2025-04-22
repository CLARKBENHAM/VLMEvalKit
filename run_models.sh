#!/usr/bin/env bash
mkdir -p logs
LOGFILE="logs/vlmevalkit_$(date +%Y%m%dT%H%M%S).log"

# seperate installed versions
LMDEPLOY_PYTHON="/data2/Users/clark/miniconda3/envs/lmdeploy/bin/python"
VLMEVALKIT_PYTHON="/data2/Users/clark/miniconda3/envs/vlmevalkit/bin/python"
LMDEPLOY_BIN="/data2/Users/clark/miniconda3/envs/lmdeploy/bin/lmdeploy"

# Ran into ray problems when using --tp
# Create custom temp directory
CUSTOM_RAY_DIR="/tmp/ray_custom_$(whoami)_$(date +%s)"
mkdir -p "$CUSTOM_RAY_DIR"

# Set environment variables that Ray will use
export RAY_OVERRIDE_TEMP_DIR="$CUSTOM_RAY_DIR"
export RAY_TMPDIR="$CUSTOM_RAY_DIR"
export RAY_OVERRIDE_REDIS_PORT=6399  # Default is 6379
export RAY_OVERRIDE_DASHBOARD_PORT=8299  # Default is 8265
export RAY_START_HEAD=1

unset RAY_ADDRESS
unset RAY_JOB_ID
unset RAY_RUNTIME_ENV_PRECREATE_FAIL_BEHAVIOR
unset RAY_RUNTIME_ENV_LOCAL_CACHE_SIZE_GB

export RAY_TEMP_DIR="/tmp/ray_$(whoami)_$(date +%s)"
export RAY_PLASMA_DIR="$RAY_TEMP_DIR/plasma"
mkdir -p "$RAY_TEMP_DIR" "$RAY_PLASMA_DIR"

# Start your own Ray instance with custom ports
# ray start --head \
#     --port 6390 \
#     --gcs-server-port 6391 \
#     --object-manager-port 8086 \
#     --node-manager-port 8087 \
#     --dashboard-port 8299 \
#     --temp-dir "$RAY_TEMP_DIR" \
#     --plasma-directory "$RAY_PLASMA_DIR" \
#     --redis-password "your_secret_$(whoami)"

# Set environment variables to tell lmdeploy to use your Ray instance

for MODEL in \
	meta-llama/Llama-3.2-11B-Vision \
    meta-llama/Llama-3.2-11B-Vision-Instruct \
    meta-llama/Llama-3.2-90B-Vision \
    meta-llama/Llama-3.2-90B-Vision-Instruct \
	google/gemma-3-27b-it \
	google/gemma-3-27b-it-qat-q4_0-gguf \
	Qwen/Qwen2.5-VL-72B-Instruct \
	Qwen/Qwen2.5-VL-32B-Instruct \
	Qwen/Qwen2.5-VL-7B-Instruct \
	microsoft/Phi-3.5-vision-instruct \
	microsoft/Phi-3.5-vision \
	AI4Chem/ChemVLM-26B \
	stepfun-ai/GOT-OCR2_0 \
	OpenGVLab/InternVL2-8B \
	openbmb/MiniCPM-Llama3-V-2_5 \
	Qwen/Qwen2.5-Omni-7B
do
# GOT-OCR2_0 and below Unsuported by LMDeploy, we'll see if run
  echo
  echo "=== $MODEL @ $(date +'%Y-%m-%dT%H:%M:%S') ===" | tee -a "$LOGFILE"

	# serve api_server "$MODEL" \ # or serve proxy so can use multiple GPUs?
	CUDA_VISIBLE_DEVICES=4,5,6,7 $LMDEPLOY_BIN \
  	serve api_server "$MODEL" \
  	--model-name "${MODEL##*/}" \
	--tp 4
	# --tp had Ray issues, and must devide number of heads
	# --session-len 256000

	# --server-port 23339
	# &
  #
	# export LMDEPLOY_API_BASE=http://localhost:23339/ # causes errors with invalid path
	# should be http://0.0.0.0:23333 ?

	SERVER_PID=$!
	until lsof -i:23333; do echo "waiting for server"; sleep 1; done

	echo "Starting run $(date +'%Y-%m-%dT%H:%M:%S')" | tee -a "$LOGFILE"

  # 4) Run your eval
  TOKENIZERS_PARALLELISM=True $VLMEVALKIT_PYTHON run.py \
    --data gdt_vlmeval \
    --model lmdeploy \
    --api-nproc 1 \
    --retry 3 \
    --verbose \
    --reuse \
  2>&1 | tee -a "$LOGFILE"

  # 5) Tear it down
  echo "Killing LMDeploy server (pid=$SERVER_PID)" | tee -a "$LOGFILE"
  kill $SERVER_PID

done