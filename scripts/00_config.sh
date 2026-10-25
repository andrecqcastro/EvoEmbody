#!/bin/bash
#
# Settings shared by the setup and run scripts. Sourced, not run directly.
# Every value can be overridden from the environment:
#
#   MODEL_PROFILE=gemma bash scripts/03_run_evolution.sh --max_cycles 20
#   TASK_THINKING=off bash scripts/03_run_evolution.sh --max_cycles 20
#
# When META_MODEL_DIR and TASK_MODEL_DIR are the same checkpoint, one server
# serves both roles. Otherwise META_GPU/TASK_GPU choose the GPU for each role;
# setting them equal co-locates both servers on one GPU.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_VLLM="$PROJECT_DIR/env_vllm"          # vLLM server environment
ENV_PIPE="$PROJECT_DIR/env_pipeline"      # evolution and ALFWorld environment
CACHE_DIR="$PROJECT_DIR/cache"
MODEL_ROOT="$PROJECT_DIR/models"

# vLLM's --max-model-len is the total of prompt and generation; each role
# declares both halves and is served with their sum. Kept identical across
# model pairs.
META_PROMPT_CONTEXT="${META_PROMPT_CONTEXT:-8192}"
META_MAX_TOKENS="${META_MAX_TOKENS:-32768}"
TASK_PROMPT_CONTEXT="${TASK_PROMPT_CONTEXT:-4096}"
TASK_MAX_TOKENS="${TASK_MAX_TOKENS:-16384}"
TASK_THINKING_BUDGET="${TASK_THINKING_BUDGET:-1024}"

# Model pair. META_MODEL_DIR/TASK_MODEL_DIR/*_THINKING override the profile.
MODEL_PROFILE="${MODEL_PROFILE:-qwen}"

case "$MODEL_PROFILE" in
    qwen)
        DEFAULT_META_MODEL_DIR="$MODEL_ROOT/Qwen/Qwen3.8-27B"
        DEFAULT_TASK_MODEL_DIR="$MODEL_ROOT/Qwen/Qwen3.5-4B"
        ;;
    gemma)
        DEFAULT_META_MODEL_DIR="$MODEL_ROOT/google/gemma-4-31B-it"
        DEFAULT_TASK_MODEL_DIR="$MODEL_ROOT/google/gemma-4-26B-A4B-it"
        ;;
    *)
        echo "ERROR: unknown MODEL_PROFILE '$MODEL_PROFILE'. Known: qwen, gemma." >&2
        exit 1
        ;;
esac

META_MODEL_DIR="${META_MODEL_DIR:-$DEFAULT_META_MODEL_DIR}"
TASK_MODEL_DIR="${TASK_MODEL_DIR:-$DEFAULT_TASK_MODEL_DIR}"
META_THINKING="${META_THINKING:-on}"
TASK_THINKING="${TASK_THINKING:-on}"
# Optional, passed to `vllm serve --quantization` when set (e.g. fp8).
META_QUANTIZATION="${META_QUANTIZATION:-}"
TASK_QUANTIZATION="${TASK_QUANTIZATION:-}"

META_PORT="${META_PORT:-8080}"
TASK_PORT="${TASK_PORT:-8081}"

# GPU for each role. Both default to GPU 0 (co-located). With two or more GPUs,
# set TASK_GPU to a different index to give each role its own device.
META_GPU="${META_GPU:-0}"
TASK_GPU="${TASK_GPU:-0}"
# Memory fraction per server when each role has its own GPU.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
# Memory fractions when both servers share one GPU. The split that works
# depends on the GPU's total memory; verify with PREFLIGHT_ONLY=1 on a new
# machine. The Gemma pair needs two GPUs.
META_GPU_MEM_UTIL="${META_GPU_MEM_UTIL:-0.50}"
TASK_GPU_MEM_UTIL="${TASK_GPU_MEM_UTIL:-0.15}"

# The served name identifies the checkpoint and is stored in the lineage
# archive, so scores from different models are never mixed.
META_MODEL_NAME="$(basename "$META_MODEL_DIR" | tr 'A-Z' 'a-z')"
TASK_MODEL_NAME="$(basename "$TASK_MODEL_DIR" | tr 'A-Z' 'a-z')"
META_MAX_MODEL_LEN=$((META_PROMPT_CONTEXT + META_MAX_TOKENS))
TASK_MAX_MODEL_LEN=$((TASK_PROMPT_CONTEXT + TASK_MAX_TOKENS))

# One checkpoint for both roles: a single server whose window covers both.
if [ "$META_MODEL_DIR" = "$TASK_MODEL_DIR" ]; then
    SHARED_SERVER=1
    TASK_PORT="$META_PORT"
    TASK_GPU="$META_GPU"
    TASK_MODEL_NAME="$META_MODEL_NAME"
    [ "$TASK_MAX_MODEL_LEN" -gt "$META_MAX_MODEL_LEN" ] && META_MAX_MODEL_LEN="$TASK_MAX_MODEL_LEN"
    TASK_MAX_MODEL_LEN="$META_MAX_MODEL_LEN"
else
    SHARED_SERVER=0
fi

# Two different checkpoints on the same GPU.
if [ "$SHARED_SERVER" = "0" ] && [ "$META_GPU" = "$TASK_GPU" ]; then
    SAME_DEVICE=1
else
    SAME_DEVICE=0
fi

# Caches live under the project directory.
export HF_HOME="$CACHE_DIR/huggingface"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
# Do not set PIP_NO_USER: pip reads it without inverting the boolean.
export PIP_USER=0
export PYTHONNOUSERSITE=1
export TORCH_HOME="$CACHE_DIR/torch"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export CONDA_PKGS_DIRS="$CACHE_DIR/conda_pkgs"
export TMPDIR="$CACHE_DIR/tmp"
export FLASHINFER_WORKSPACE_BASE="$CACHE_DIR/flashinfer"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export ALFWORLD_DATA="$PROJECT_DIR/alfworld_data"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS" "$TMPDIR" \
         "$FLASHINFER_WORKSPACE_BASE" "$TORCH_EXTENSIONS_DIR" \
         "$ALFWORLD_DATA" "$PROJECT_DIR/server_logs"

# Optional .env file with KEY=value lines.
if [ -f "$PROJECT_DIR/.env" ]; then
    if grep -qE '^[A-Za-z_][A-Za-z0-9_]*[[:space:]]+=' "$PROJECT_DIR/.env"; then
        echo "ERROR: there is a space before '=' in .env. Fix it to KEY=value." >&2
        exit 1
    fi
    set -a; source "$PROJECT_DIR/.env"; set +a
    echo ".env loaded."
fi

# Activates a conda environment and fails loudly if it did not take effect.
activate_env() {
    conda activate "$1"
    if [ "$CONDA_PREFIX" != "$1" ]; then
        echo "ERROR: conda activate did not activate $1 (CONDA_PREFIX=$CONDA_PREFIX)." >&2
        exit 1
    fi
}
