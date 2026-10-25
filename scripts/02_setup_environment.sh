#!/bin/bash
#
# Builds the two conda environments and downloads the ALFWorld data.
#
#   bash scripts/02_setup_environment.sh           # everything
#   bash scripts/02_setup_environment.sh server    # only the vLLM environment
#   bash scripts/02_setup_environment.sh pipeline  # only the pipeline + data
#
# Two environments, because vLLM pins torch and transformers versions that
# conflict with alfworld/textworld.
#
# CUDA comes from the conda cuda-nvcc package, whose activation hook puts nvcc
# on PATH for every child process, including vLLM's EngineCore.
set -e

STAGE="${1:-all}"
case "$STAGE" in
    all|server|pipeline) ;;
    *) echo "ERROR: unknown stage '$STAGE'. Use: all | server | pipeline" >&2; exit 1 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_config.sh"
cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "$STAGE" != "pipeline" ]; then
echo "=========================================="
echo "1/3 - vLLM server environment ($ENV_VLLM)"
echo "=========================================="
# The server environment must be Python 3.10.
if [ -d "$ENV_VLLM" ] && ! "$ENV_VLLM/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null; then
    echo "ERROR: $ENV_VLLM exists but is not Python 3.10." >&2
    echo "       Move it aside and re-run:  mv '$ENV_VLLM' '$ENV_VLLM.old'" >&2
    exit 1
fi
[ -d "$ENV_VLLM" ] || conda create --prefix "$ENV_VLLM" python=3.10 -y

# cuda-nvcc pulls the conda gcc/gxx toolchain its own activation hook needs.
conda install --prefix "$ENV_VLLM" -c nvidia cuda-toolkit cuda-nvcc -y

activate_env "$ENV_VLLM"
"$ENV_VLLM/bin/python" -s -m pip install --upgrade pip wheel
"$ENV_VLLM/bin/python" -s -m pip install -r scripts/requirements_vllm.txt
"$ENV_VLLM/bin/python" -s -m pip check

# Link the system driver's libcuda.so into the environment (x86_64 or aarch64).
SYSTEM_LIBCUDA="/usr/lib/$(uname -m)-linux-gnu/libcuda.so"
if [ -f "$SYSTEM_LIBCUDA" ]; then
    ln -sf "$SYSTEM_LIBCUDA" "$ENV_VLLM/lib/libcuda.so"
    echo "libcuda.so linked at $ENV_VLLM/lib/libcuda.so"
else
    echo "WARNING: $SYSTEM_LIBCUDA not found."
fi

echo "GATE: CUDA, imports, model architecture and serving flags"
META_MODEL_DIR="$META_MODEL_DIR" TASK_MODEL_DIR="$TASK_MODEL_DIR" python <<'PY'
import json, os, shutil, subprocess, sys

nvcc = shutil.which("nvcc")
if not nvcc:
    sys.exit("FAIL: nvcc is not on PATH after activation; the conda cuda-nvcc package did not land.")
print("OK  nvcc:", nvcc)

import torch; print("OK  torch", torch.__version__)
import vllm;  print("OK  vllm", vllm.__version__)
from vllm.entrypoints.cli.main import main  # noqa: F401
print("OK  vllm CLI entrypoint")

import flashinfer.comm  # noqa: F401
import flashinfer; print("OK  flashinfer", flashinfer.__version__)
import flashinfer_cubin  # noqa: F401
print("OK  flashinfer_cubin (prebuilt kernels: the sampler does no JIT)")

from vllm.model_executor.models.registry import ModelRegistry
supported = ModelRegistry.get_supported_archs()
for role in ("META_MODEL_DIR", "TASK_MODEL_DIR"):
    directory = os.environ[role]
    if not os.path.isdir(directory):
        print(f"WARNING: {role} not downloaded yet ({directory}). Set {role} to an existing "
              "checkpoint directory, or download one there yourself (e.g. `hf download`).")
        continue
    for arch in json.load(open(os.path.join(directory, "config.json")))["architectures"]:
        if arch not in supported:
            sys.exit(f"FAIL: vllm {vllm.__version__} does not register {arch} ({role}).")
        print(f"OK  {arch} registered ({role})")

# `vllm serve --help` prints only a summary; the full flag list needs --help=all.
help_all = subprocess.run(["vllm", "serve", "--help=all"], capture_output=True, text=True).stdout
missing = [f for f in ("--enable-auto-tool-choice", "--tool-call-parser", "--reasoning-parser")
           if f not in help_all]
if missing:
    sys.exit(f"FAIL: this vLLM build does not have {missing}.")
for parser in ("qwen3_xml", "qwen3"):
    if parser not in help_all:
        print(f"WARNING: parser '{parser}' does not appear in --help=all.")
print("OK  --enable-auto-tool-choice / --tool-call-parser qwen3_xml / --reasoning-parser qwen3")

# The task model's thinking budget depends on this per-request parameter.
from vllm.sampling_params import SamplingParams
field_names = (getattr(SamplingParams, "model_fields", None)
              or getattr(SamplingParams, "__struct_fields__", None)
              or getattr(SamplingParams, "__fields__", None) or ())
if "thinking_token_budget" not in field_names:
    print("WARNING: this vLLM build has no thinking_token_budget; the task model will think "
          "without a budget and the run will record the fallback.")
else:
    print("OK  thinking_token_budget supported (moderate thinking for the task model)")
PY
conda deactivate
fi  # STAGE != pipeline

if [ "$STAGE" = "server" ]; then
    echo
    echo "Server environment ready. The pipeline environment was left untouched."
    exit 0
fi

echo "=========================================="
echo "2/3 - Pipeline environment ($ENV_PIPE)"
echo "=========================================="
[ -d "$ENV_PIPE" ] || conda create --prefix "$ENV_PIPE" python=3.12 -y
activate_env "$ENV_PIPE"
"$ENV_PIPE/bin/python" -s -m pip install --upgrade pip setuptools wheel

# textworld's installer downloads Inform7, which has no aarch64 build. Install a
# patched textworld first (see scripts/patch_textworld_for_arm.py).
"$ENV_PIPE/bin/python" scripts/patch_textworld_for_arm.py "$ENV_PIPE/bin/python"
"$ENV_PIPE/bin/python" -s -m pip install -r requirements.txt
python -c "import alfworld, textworld, openai, langchain_openai, yaml; print('OK  pipeline imports')"

echo "=========================================="
echo "3/3 - ALFWorld data ($ALFWORLD_DATA)"
echo "=========================================="
# Keep the data inside the project instead of ~/.cache/alfworld.
if [ -d "$ALFWORLD_DATA/json_2.1.1" ]; then
    echo "Data already present."
else
    "$ENV_PIPE/bin/alfworld-download" --data-dir "$ALFWORLD_DATA"
fi

echo "Checking the games the evolution will actually play..."
python <<'PY'
from src.benchmarks.alfworld.alfworld_archive import TYPE_ORDER
from src.eval.lineage_archive import FIRST_STAGE_SIZE, build_stage_plan
from src.benchmarks.alfworld.alfworld_env import games_by_type
from src.benchmarks.alfworld.generate_alfworld_predictions import collect_game_files

wanted = games_by_type(collect_game_files("eval_in_distribution"))
counts = {t: len(wanted.get(t, [])) for t in TYPE_ORDER}
total = sum(counts.values())
print("games per type:", counts, "| total:", total)
missing = [t for t in TYPE_ORDER if not wanted.get(t)]
assert not missing, f"no games found for task types: {missing}"
assert total >= FIRST_STAGE_SIZE, f"need at least {FIRST_STAGE_SIZE} games, only {total} exist"
print("OK  the pool covers", build_stage_plan(total))
PY

echo
echo "=========================================="
echo "Environments ready."
for d in "$META_MODEL_DIR" "$TASK_MODEL_DIR"; do
    [ -d "$d" ] && echo "Model present: $d" || echo "WARNING: $d is missing. Download the checkpoint there."
done
echo "Next: bash scripts/03_run_evolution.sh --max_cycles 20"
echo "=========================================="
