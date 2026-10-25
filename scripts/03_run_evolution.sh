#!/bin/bash
#
# Starts the vLLM server(s) and runs the evolution. Run it inside tmux/screen
# or with nohup so it survives a terminal disconnect.
#
#   bash scripts/03_run_evolution.sh --max_cycles 20
#   bash scripts/03_run_evolution.sh --max_tokens 50000000 --max_cycles none
#   bash scripts/03_run_evolution.sh --resume --max_cycles 10
#   PREFLIGHT_ONLY=1 bash scripts/03_run_evolution.sh
#
# PREFLIGHT_ONLY=1 starts both servers, runs scripts/preflight.py (which asks
# each server for a real generation), then shuts everything down.
#
# EVOLUTION_ENTRYPOINT selects the Python script started once the servers are
# up (default run_evolution.py), e.g. EVOLUTION_ENTRYPOINT=run_final_test.py.
# All other arguments are passed through to it; --help lists them.
#
# Model pair, thinking and window sizes come from scripts/00_config.sh:
#
#   MODEL_PROFILE=qwen TASK_THINKING=off bash scripts/03_run_evolution.sh --max_cycles 20
#
# GPU assignment:
#
#   META_GPU=0 TASK_GPU=1 bash scripts/03_run_evolution.sh ...   # one role per GPU
#
# Co-location (META_GPU == TASK_GPU) splits one GPU between the two servers
# with META_GPU_MEM_UTIL/TASK_GPU_MEM_UTIL. The Qwen pair fits on one large
# GPU; the Gemma pair should use two GPUs.
#
# When running several instances at once, pass --max_workers explicitly: the
# automatic sizing below cannot see the other runs.
#
# The archive is rewritten after every stage, so an interrupted run resumes with
#   bash scripts/03_run_evolution.sh --resume
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/00_config.sh"
cd "$PROJECT_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
META_LOG="$PROJECT_DIR/server_logs/vllm_meta_${RUN_ID}.log"
TASK_LOG="$PROJECT_DIR/server_logs/vllm_task_${RUN_ID}.log"
RESERVE_CORES=6        # cores kept for the server(s) and the OS
PER_WORKER_MB=1500     # RAM per episode worker process
OVERSUBSCRIBE=2        # workers per remaining core: they mostly wait on HTTP

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=========================================="
echo "1. Sizing against this machine's resources"
echo "=========================================="
activate_env "$ENV_PIPE"

PARAMS=$(META_MODEL_DIR="$META_MODEL_DIR" TASK_MODEL_DIR="$TASK_MODEL_DIR" \
         META_MAX_MODEL_LEN="$META_MAX_MODEL_LEN" TASK_MAX_MODEL_LEN="$TASK_MAX_MODEL_LEN" \
         SHARED_SERVER="$SHARED_SERVER" META_GPU="$META_GPU" TASK_GPU="$TASK_GPU" \
         GPU_MEM_UTIL="$GPU_MEM_UTIL" META_GPU_MEM_UTIL="$META_GPU_MEM_UTIL" \
         TASK_GPU_MEM_UTIL="$TASK_GPU_MEM_UTIL" \
         RESERVE_CORES="$RESERVE_CORES" PER_WORKER_MB="$PER_WORKER_MB" \
         OVERSUBSCRIBE="$OVERSUBSCRIBE" python <<'PY'
import contextlib, json, math, os, subprocess, sys

log = lambda m: print(m, file=sys.stderr)
GIB = 1024 ** 3

# ---- episode workers -------------------------------------------------------
# Each stage only plays the games it adds, so peak concurrency is the largest
# step between consecutive stages.
from src.benchmarks.alfworld.alfworld_env import games_by_type
from src.benchmarks.alfworld.generate_alfworld_predictions import collect_game_files
from src.eval.lineage_archive import build_stage_plan

# AlfredTWEnv prints progress to stdout, which would end up in $PARAMS.
with contextlib.redirect_stdout(sys.stderr):
    pool = games_by_type(collect_game_files("eval_in_distribution"))
total_pool_size = sum(len(v) for v in pool.values())
stage_plan = build_stage_plan(total_pool_size)
deltas, previous = {}, 0
for name, size in stage_plan:
    deltas[name] = size - previous
    previous = size
ceiling = max(deltas.values())

cores = len(os.sched_getaffinity(0))
cpu_limit = max(1, (cores - int(os.environ["RESERVE_CORES"])) * int(os.environ["OVERSUBSCRIBE"]))


def cgroup_limit_mb():
    for path in ("/sys/fs/cgroup/memory.max",                     # cgroup v2
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v1
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        value = int(raw)
        if 0 < value < (1 << 62):   # v1 reports a sentinel near 2**63 when unset
            return value // 1024 ** 2
    return 0


# Prefer the cgroup limit; /proc/meminfo covers the whole machine. On
# unified-memory systems this is measured before vLLM claims its share, so
# lower PER_WORKER_MB or pass --max_workers if workers get OOM-killed.
mem_mb, source = cgroup_limit_mb(), "cgroup limit"
if not mem_mb:
    mem_mb = int(next(l.split()[1] for l in open("/proc/meminfo")
                      if l.startswith("MemAvailable:"))) // 1024
    source = "/proc/meminfo (whole machine)"

ram_limit = max(1, int(0.75 * mem_mb / int(os.environ["PER_WORKER_MB"])))
workers = min(ceiling, cpu_limit, ram_limit)

log(f"  new games per stage : {deltas}")
log(f"  stage ceiling       : {ceiling}")
log(f"  CPU limit           : {cpu_limit}  (({cores} - {os.environ['RESERVE_CORES']}) x {os.environ['OVERSUBSCRIBE']})")
log(f"  RAM limit           : {ram_limit}  ({mem_mb} MB / {os.environ['PER_WORKER_MB']} MB) [{source}]")
log(f"  => WORKERS          : {workers}")
waves = {s: math.ceil(n / workers) for s, n in deltas.items()}
log(f"  waves per stage     : {waves}"
    f"{'' if max(waves.values()) == 1 else '   <-- >1 means a stage runs in batches'}")

# ---- the server(s) ---------------------------------------------------------
shared = os.environ["SHARED_SERVER"] == "1"
meta_device, task_device = os.environ["META_GPU"], os.environ["TASK_GPU"]
same_device = (not shared) and meta_device == task_device


def gpu_bytes(device):
    name, total = [x.strip() for x in subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=name,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        check=True).stdout.strip().split(",")]
    try:
        return name, int(total) * 1024 ** 2
    except ValueError:
        # Unified-memory systems (e.g. DGX Spark) report N/A; GPU and CPU share
        # system RAM there.
        mem_total_kb = int(next(l.split()[1] for l in open("/proc/meminfo")
                                 if l.startswith("MemTotal:")))
        return name, mem_total_kb * 1024


def total_weight_bytes(model_dir):
    # On-disk size, which overestimates weights quantized at load time.
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        return int(json.load(open(index_path))["metadata"]["total_size"])
    return os.path.getsize(os.path.join(model_dir, "model.safetensors"))


def kv_cost_model(config):
    """
    (kv_per_token, state_per_seq, note) for the supported architectures.
    Qwen3.5+ hybrids keep a constant-size state per sequence for their linear
    attention layers. Gemma's sliding-window layers are sized as full
    attention, which errs on the safe side.
    """
    layer_types = config.get("layer_types")
    if layer_types is None:
        n = config["num_hidden_layers"]
        kv_per_token = 2 * config["num_key_value_heads"] * config["head_dim"] * n * 2
        return kv_per_token, 0, "uniform full attention"

    if "linear_num_value_heads" in config:
        n_full, n_linear = layer_types.count("full_attention"), layer_types.count("linear_attention")
        kv_per_token = 2 * config["num_key_value_heads"] * config["head_dim"] * n_full * 2
        state_per_seq = n_linear * (
            config["linear_num_value_heads"] * config["linear_key_head_dim"]
            * config["linear_value_head_dim"] * 4
            + (2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
               + config["linear_num_value_heads"] * config["linear_value_head_dim"])
            * config["linear_conv_kernel_dim"] * 2)
        return kv_per_token, state_per_seq, f"hybrid linear+full ({n_full} full, {n_linear} linear)"

    kv_heads = max(config["num_key_value_heads"], config.get("num_global_key_value_heads") or 0)
    head_dim = max(config["head_dim"], config.get("global_head_dim") or 0)
    kv_per_token = 2 * kv_heads * head_dim * len(layer_types) * 2
    n_full, n_sliding = layer_types.count("full_attention"), layer_types.count("sliding_attention")
    return kv_per_token, 0, f"hybrid full+sliding ({n_full} full, {n_sliding} sliding, sized as full attention)"


def size_server(role, model_dir, target, desired_seqs, device, util):
    config = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    weights = total_weight_bytes(model_dir)
    kv_per_token, state_per_seq, note = kv_cost_model(config)

    name, gpu = gpu_bytes(device)
    budget = gpu * util - weights - 4 * GIB          # 4 GiB for activations

    if state_per_seq:
        max_seqs = max(1, min(desired_seqs, int(budget * 0.45 // state_per_seq)))
    else:
        max_seqs = desired_seqs
    max_len = min(target, int((budget - max_seqs * state_per_seq) // kv_per_token))

    log(f"  [{role}] device {device}: {name}, {gpu / GIB:.1f} GiB, util {util} ({note})")
    log(f"  [{role}] weights {weights / GIB:.1f} GiB | cache budget {budget / GIB:.1f} GiB "
        f"| KV {kv_per_token / 1024:.0f} KiB/token | state {state_per_seq / 1024**2:.0f} MiB/seq")
    log(f"  [{role}] => MAX_NUM_SEQS {max_seqs}, MAX_MODEL_LEN {max_len} (target {target})")

    if max_len < 8192:
        sys.exit(f"[{role}] only {max_len} tokens of context fit on this GPU/util combination. "
                 f"Raise *_GPU_MEM_UTIL or put this role on its own GPU.")
    if max_seqs < desired_seqs:
        log(f"  [{role}] {desired_seqs - max_seqs} of {desired_seqs} requests will queue.")
    return max_seqs, max_len


# The meta-agents run one at a time; each episode worker holds one task sequence.
META_SEQS = 4
if shared:
    util = float(os.environ["GPU_MEM_UTIL"])
    seqs, length = size_server("shared", os.environ["META_MODEL_DIR"],
                               int(os.environ["META_MAX_MODEL_LEN"]),
                               workers + META_SEQS, meta_device, util)
    meta_seqs = task_seqs = seqs
    meta_len = task_len = length
else:
    meta_util = float(os.environ["META_GPU_MEM_UTIL"] if same_device else os.environ["GPU_MEM_UTIL"])
    task_util = float(os.environ["TASK_GPU_MEM_UTIL"] if same_device else os.environ["GPU_MEM_UTIL"])
    meta_seqs, meta_len = size_server("meta", os.environ["META_MODEL_DIR"],
                                      int(os.environ["META_MAX_MODEL_LEN"]), META_SEQS,
                                      meta_device, meta_util)
    task_seqs, task_len = size_server("task", os.environ["TASK_MODEL_DIR"],
                                      int(os.environ["TASK_MAX_MODEL_LEN"]), workers,
                                      task_device, task_util)

print(f"WORKERS={workers}")
print(f"META_MAX_NUM_SEQS={meta_seqs}")
print(f"META_MAX_MODEL_LEN={meta_len}")
print(f"TASK_MAX_NUM_SEQS={task_seqs}")
print(f"TASK_MAX_MODEL_LEN={task_len}")
PY
) || { echo "ERROR: sizing failed (see above)." >&2; exit 1; }
eval "$PARAMS"
conda deactivate

# More workers than the task server can serve concurrently would only queue.
if [ "$TASK_MAX_NUM_SEQS" -lt "$WORKERS" ]; then
    echo "Capping workers from $WORKERS to the task server's concurrency ($TASK_MAX_NUM_SEQS)."
    WORKERS="$TASK_MAX_NUM_SEQS"
fi

EVOL_PID=""
TASK_PID=""
META_PID=""
# Each server and the evolution run in their own process group (setsid), so
# the whole group, including vLLM's EngineCore child, can be stopped.
kill_group() {
    local pgid="$1"
    [ -n "$pgid" ] || return 0
    kill -TERM -- "-$pgid" 2>/dev/null || true
}
# Also walk the process tree, for descendants that left the group.
kill_descendants() {
    local pid="$1"
    [ -n "$pid" ] || return 0
    local child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        kill_descendants "$child"
    done
    kill -KILL "$pid" 2>/dev/null || true
}
shutdown_all() {
    kill_group "$EVOL_PID"
    kill_group "$TASK_PID"
    kill_group "$META_PID"
    sleep 3
    kill -KILL -- "-$EVOL_PID" 2>/dev/null || true
    kill -KILL -- "-$TASK_PID" 2>/dev/null || true
    kill -KILL -- "-$META_PID" 2>/dev/null || true
    kill_descendants "$EVOL_PID"
    kill_descendants "$TASK_PID"
    kill_descendants "$META_PID"
}
trap shutdown_all EXIT

# Ctrl-C or SIGTERM stops the evolution and the servers. State is already in
# the archive, so the run can be continued with --resume.
on_interrupt() {
    echo ""
    echo "=========================================="
    echo "Interrupted at $(date '+%F %T'), stopping."
    echo "To continue: bash scripts/03_run_evolution.sh --resume"
    echo "=========================================="
    kill -INT "$EVOL_PID" 2>/dev/null || true
    sleep 5
    shutdown_all
    exit 0
}
trap on_interrupt SIGINT SIGTERM

echo "=========================================="
echo "2. Starting the vLLM server(s)"
echo "=========================================="
activate_env "$ENV_VLLM"

SYSTEM_LIBCUDA="/usr/lib/$(uname -m)-linux-gnu/libcuda.so"
if [ ! -e "$ENV_VLLM/lib/libcuda.so" ] && [ -f "$SYSTEM_LIBCUDA" ]; then
    ln -sf "$SYSTEM_LIBCUDA" "$ENV_VLLM/lib/libcuda.so"
fi

# Tool calling is required by the meta-agents. The reasoning parser keeps
# <think> blocks out of `content` and enables thinking_token_budget. Parsers
# are chosen per role from the checkpoint family: Qwen3.5+ emits XML tool
# calls (qwen3_xml), Gemma 4 uses the gemma4 parsers.
HELP_ALL="$(vllm serve --help=all 2>&1 || true)"

parser_flags_for() {   # model_dir
    local model_dir="$1" family tool_call_parser reasoning_parser flags=""
    family="$(tr 'A-Z' 'a-z' <<< "$model_dir")"
    case "$family" in
        *gemma*) tool_call_parser=gemma4;     reasoning_parser=gemma4 ;;
        *qwen*)  tool_call_parser=qwen3_xml;  reasoning_parser=qwen3  ;;
        *)
            echo "ERROR: could not tell the model family of '$model_dir' (expected" \
                 "'qwen' or 'gemma' in the path) to pick its tool-call/reasoning parser." >&2
            exit 1
            ;;
    esac
    grep -q -- "--enable-auto-tool-choice" <<< "$HELP_ALL" && flags="$flags --enable-auto-tool-choice"
    grep -q "$tool_call_parser" <<< "$HELP_ALL" && flags="$flags --tool-call-parser $tool_call_parser"
    grep -q -- "--reasoning-parser" <<< "$HELP_ALL" && flags="$flags --reasoning-parser $reasoning_parser"
    echo "$flags"
}

META_SERVE_FLAGS="$(parser_flags_for "$META_MODEL_DIR")"
TASK_SERVE_FLAGS="$(parser_flags_for "$TASK_MODEL_DIR")"
[ -n "$META_QUANTIZATION" ] && META_SERVE_FLAGS="$META_SERVE_FLAGS --quantization $META_QUANTIZATION"
[ -n "$TASK_QUANTIZATION" ] && TASK_SERVE_FLAGS="$TASK_SERVE_FLAGS --quantization $TASK_QUANTIZATION"
echo "Meta serving flags:$META_SERVE_FLAGS"
echo "Task serving flags:$TASK_SERVE_FLAGS"
if [ -z "$META_SERVE_FLAGS" ] || [ -z "$TASK_SERVE_FLAGS" ]; then
    echo "ERROR: no tool-calling flag was detected; the preflight would fail anyway." >&2
    exit 1
fi

# thinking_token_budget is applied by the V1 model runner.
export VLLM_USE_V2_MODEL_RUNNER=0

serve() {   # role model_dir served_name port device max_len max_seqs util log serve_flags
    echo "  $1: $(basename "$2") on GPU $5, port $4, window $6, $7 concurrent"
    setsid env CUDA_VISIBLE_DEVICES="$5" vllm serve "$2" \
        --served-model-name "$3" \
        --port "$4" \
        --trust-remote-code \
        --max-model-len "$6" \
        --max-num-seqs "$7" \
        --gpu-memory-utilization "$8" \
        ${10} > "$9" 2>&1 &
}

if [ "$SAME_DEVICE" = "1" ]; then
    META_UTIL="$META_GPU_MEM_UTIL"
    TASK_UTIL="$TASK_GPU_MEM_UTIL"
else
    META_UTIL="$GPU_MEM_UTIL"
    TASK_UTIL="$GPU_MEM_UTIL"
fi

wait_for_server() {
    local role="$1" port="$2" pid="$3" log="$4"
    echo "Waiting for the $role server on port $port..."
    while ! curl -f -s "http://127.0.0.1:$port/health" > /dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "CRITICAL: the $role server died during startup. End of $log:" >&2
            tail -40 "$log" >&2
            exit 1
        fi
        sleep 10
    done
    echo "$role server online."
    grep -iE "KV cache|GPU blocks|Loading weights" "$log" | tail -3 || true
}

serve meta "$META_MODEL_DIR" "$META_MODEL_NAME" "$META_PORT" "$META_GPU" \
      "$META_MAX_MODEL_LEN" "$META_MAX_NUM_SEQS" "$META_UTIL" "$META_LOG" "$META_SERVE_FLAGS"
META_PID=$!

if [ "$SAME_DEVICE" = "1" ] && [ "$SHARED_SERVER" != "1" ]; then
    # Co-located servers start one after the other: vLLM sizes its cache from
    # the memory free at startup, so starting both at once is a race.
    wait_for_server "meta" "$META_PORT" "$META_PID" "$META_LOG"

    # On unified-memory systems the page cache filled while loading the meta
    # checkpoint is not reported as free GPU memory, and the task server can
    # fail its first allocation (vllm-project/vllm#35313). Touching a buffer of
    # that size forces the kernel to evict the cache without root access.
    activate_env "$ENV_PIPE"
    python - <<'PY'
import ctypes
import os


def cached_bytes():
    for line in open("/proc/meminfo"):
        if line.startswith("Cached:"):
            return int(line.split()[1]) * 1024
    return 0


os.system("sync")
size = cached_bytes()
if size > 512 * 1024 * 1024:
    buf = bytearray(size)
    ctypes.memset((ctypes.c_char * size).from_buffer(buf), 1, size)
    del buf
    print(f"[cache pressure] evicted {size / 1024**3:.1f} GiB of page cache "
          f"before starting the task server.")
else:
    print("[cache pressure] skipped: little reclaimable cache present.")
PY
    activate_env "$ENV_VLLM"
fi

if [ "$SHARED_SERVER" = "1" ]; then
    echo "  task: same checkpoint as meta, reusing that server (thinking $TASK_THINKING)"
    TASK_PID="$META_PID"
    TASK_LOG="$META_LOG"
else
    serve task "$TASK_MODEL_DIR" "$TASK_MODEL_NAME" "$TASK_PORT" "$TASK_GPU" \
          "$TASK_MAX_MODEL_LEN" "$TASK_MAX_NUM_SEQS" "$TASK_UTIL" "$TASK_LOG" "$TASK_SERVE_FLAGS"
    TASK_PID=$!
fi

wait_for_server "meta" "$META_PORT" "$META_PID" "$META_LOG"
[ "$SHARED_SERVER" = "1" ] || wait_for_server "task" "$TASK_PORT" "$TASK_PID" "$TASK_LOG"
conda deactivate

# The limits actually being served, read by src/config/llm_config.py. Sizing may
# have lowered *_MAX_MODEL_LEN below the configured sum.
export LOCAL_API_KEY=EMPTY
export META_BASE_URL="http://127.0.0.1:$META_PORT/v1"
export META_MODEL_ID="$META_MODEL_NAME"
export META_MAX_MODEL_LEN META_PROMPT_CONTEXT META_MAX_TOKENS META_THINKING
export TASK_BASE_URL="http://127.0.0.1:$TASK_PORT/v1"
export TASK_MODEL_ID="$TASK_MODEL_NAME"
export TASK_MAX_MODEL_LEN TASK_PROMPT_CONTEXT TASK_MAX_TOKENS TASK_THINKING TASK_THINKING_BUDGET

echo "=========================================="
echo "3. Preflight"
echo "=========================================="
activate_env "$ENV_PIPE"
if ! python scripts/preflight.py; then
    echo "ERROR: preflight failed." >&2
    exit 1
fi

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo "PREFLIGHT_ONLY=1: both servers generate replies. Stopping here."
    exit 0
fi

echo "=========================================="
echo "4. Evolution (workers=$WORKERS)"
echo "=========================================="
# Run in the background and wait, so the interrupt trap fires promptly.
setsid python "${EVOLUTION_ENTRYPOINT:-run_evolution.py}" --max_workers "$WORKERS" "$@" &
EVOL_PID=$!

set +e
wait $EVOL_PID; RC=$?
# `wait` returns 128+n when a handled signal interrupts it; wait again if the
# process is still alive.
while [ $RC -gt 128 ] && kill -0 $EVOL_PID 2>/dev/null; do wait $EVOL_PID; RC=$?; done
set -e

echo "Run finished (exit code $RC)."
exit $RC
