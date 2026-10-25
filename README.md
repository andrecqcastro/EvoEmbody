# EvoEmbody

A self-evolving framework for embodied agents built on frozen open-weight
language models. A task agent plays ALFWorld, and a meta-agent diagnoses its
failures and rewrites the agent's prompt and code, with no gradient updates.

## How it works

Each cycle:

1. **Select a parent** from the archive of all agents produced so far.
2. **Mine failures** by running the parent on training tasks until it fails.
3. **Problem Understanding** inspects the failed episodes and the agent's code
   and writes a diagnostic report.
4. **Solution Proposer** reads the report and the code and writes a mutation plan.
5. **Implementation** applies the plan, producing a child agent.
6. **Evaluate** the child on validation tasks in growing tiers. Weak children
   are discarded early.

The meta-agent roles fetch logs and code on demand through tools instead of
receiving everything in one prompt. Training, validation and test tasks are
disjoint. When the run ends, the best agent is evaluated once on the held-out
test split.

The evolving agent is defined by two files:
[src/agents/archive/agent.py](src/agents/archive/agent.py) and
[src/benchmarks/alfworld/alfworld_policy.py](src/benchmarks/alfworld/alfworld_policy.py).

## Requirements

- Linux with an NVIDIA GPU and conda.
- Two model checkpoints that vLLM can serve: a larger one for the meta-agent
  and a smaller one for the task agent.

## Setup

1. Download the checkpoints under `models/<org>/<name>`, e.g.
   `models/Qwen/Qwen3.8-27B` and `models/Qwen/Qwen3.5-4B`.
2. Build the environments and download the ALFWorld data:
   ```bash
   bash scripts/02_setup_environment.sh
   ```

## Model pairs

Select a pair with `MODEL_PROFILE` (see [scripts/00_config.sh](scripts/00_config.sh)):

| Profile | Meta-agent | Task agent |
|---|---|---|
| `qwen` (default) | Qwen3.8-27B | Qwen3.5-4B |
| `gemma` | Gemma-4-31B-it | Gemma-4-26B-A4B-it |

To use other checkpoints, set `META_MODEL_DIR` and `TASK_MODEL_DIR`.

## Running

Check that both servers start and respond:
```bash
PREFLIGHT_ONLY=1 bash scripts/03_run_evolution.sh
```

Run with a token budget or a number of cycles:
```bash
bash scripts/03_run_evolution.sh --max_tokens 50000000 --max_cycles none
bash scripts/03_run_evolution.sh --max_cycles 20
```

Resume an interrupted run:
```bash
bash scripts/03_run_evolution.sh --resume
```

Other options (`--mining_source`, `--meta_mode`, `--run_tag`, ...) are listed
with `python run_evolution.py --help`.

### GPUs

Both servers share GPU 0 by default. To give each its own GPU:
```bash
META_GPU=0 TASK_GPU=1 bash scripts/03_run_evolution.sh --max_cycles 20
```

### Sandbox (optional)

To run agent-generated code inside a Docker container:
```bash
docker build -f docker/genome_sandbox.Dockerfile -t alfworld-genome-sandbox:latest .
GENOME_SANDBOX=on bash scripts/03_run_evolution.sh --max_cycles 20
```

## Outputs

Each run writes to `runs/[<run_tag>/]run_<timestamp>/`, including the lineage
archive, a summary, a lineage graph, every generated agent, and the final test
results. Confidence intervals for the test comparison:
```bash
python scripts/bootstrap_test_ci.py --run_tag <run_tag>
```
