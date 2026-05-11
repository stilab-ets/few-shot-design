#!/bin/bash
#SBATCH --job-name=llama3-experiment
#SBATCH --partition=main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:59:59
#SBATCH --output=logs/llama3_experiment_%j.out

echo "Host: $(hostname)"

CHAT_TEMPLATE_PATH="/model_templates/llama-3-instruct.jinja"

# Set model path
MODEL_NAME="google/gemma-3-12b-it"
LOCAL_MODEL_DIR="models/google/gemma-3-12b-it"

# Download model if needed
if [ ! -d "$LOCAL_MODEL_DIR" ]; then
  echo "Downloading model $MODEL_NAME to $LOCAL_MODEL_DIR..."
  huggingface-cli download $MODEL_NAME --local-dir $LOCAL_MODEL_DIR --resume-download --exclude "*.bin"
else
  echo "Model already downloaded at $LOCAL_MODEL_DIR"
fi

# Start vLLM server in the background
vllm serve "$LOCAL_MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --generation-config vllm \
  --override-generation-config '{"temperature": 0.0}' \
  > vllm_server.log 2>&1 &

# Save PID so we can wait or kill later if needed
VLLM_PID=$!

# Health check loop — wait for vLLM to accept completions
echo "Waiting for vLLM server to fully start..."

sleep 60

# Wait until vLLM server is up
until curl -s http://localhost:8000/v1 > /dev/null; do
  echo "Waiting for vLLM API to become available..."
  sleep 2
done

echo "vLLM is ready. Starting experiment script..."
echo "Running react zs - g12"
python3 -u react_zs_g12.py

echo "Experiment finished."

kill $VLLM_PID
