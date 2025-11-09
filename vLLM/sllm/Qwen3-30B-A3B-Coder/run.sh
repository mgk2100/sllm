#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec /usr/bin/env bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

: "${MODEL_ID:?MODEL_ID is required}"
: "${IMAGE:=vllm/vllm-openai:latest}"
: "${HOST_PORT:=8000}"
: "${GPU_ID:=0}"
: "${HF_CACHE_DIR:=/home/user/.cache/huggingface}"
: "${SHM_SIZE:=10.24gb}"
: "${MAX_MODEL_LEN:=128000}"
: "${TRUST_REMOTE_CODE:=true}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${CONTAINER_NAME:=vllm-single}"
: "${API_KEY:=}"
: "${ADDITIONAL_ARGS:=}"

TRUST_FLAG=""
if [ "${TRUST_REMOTE_CODE}" = "true" ] || [ "${TRUST_REMOTE_CODE}" = "1" ]; then
  TRUST_FLAG="--trust-remote-code"
fi

API_KEY_FLAG=""
if [ -n "${API_KEY}" ]; then
  API_KEY_FLAG="--api-key ${API_KEY}"
fi

set -x

docker run \
  -itd \
  --ipc host \
  --gpus "device=${GPU_ID}" \
  --shm-size "${SHM_SIZE}" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -p "${HOST_PORT}:8000" \
  --name "${CONTAINER_NAME}" "${IMAGE}" \
  --model "${MODEL_ID}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  ${TRUST_FLAG} \
  ${API_KEY_FLAG} \
  ${ADDITIONAL_ARGS}
