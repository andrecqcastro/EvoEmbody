"""
Genome mechanics: the agent's source files stored as JSON per archive node and
executed back into an agent class and a policy module.

Which files make up a genome is supplied by the benchmark (see
src/benchmarks/<name>/genome.py). Genomes are executed straight from their JSON,
so nodes never share mutated code and the repository's sources are never
overwritten.
"""
import json
import os
import sys
import types


def build_root_genome(project_root: str, dest_path: str, genome_files: tuple) -> str:
    """Writes the pristine on-disk sources out as the root node's genome."""
    files = {}
    for key in genome_files:
        with open(os.path.join(project_root, key), "r", encoding="utf-8") as f:
            files[key] = {"content": f.read()}

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump({"files": files}, f, indent=2, ensure_ascii=False)
    print(f"Root genome written to {dest_path} ({', '.join(files)})")
    return dest_path


def read_genome(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def source_of(genome: dict, key: str) -> str:
    """A genome file's content by path, falling back to a basename match."""
    files = genome.get("files", {})
    if key in files:
        return files[key]["content"]
    basename = os.path.basename(key)
    for stored_key, entry in files.items():
        if os.path.basename(stored_key) == basename:
            return entry["content"]
    raise KeyError(f"Genome is missing {key}. It holds: {list(files)}")


def load_genome(json_path: str, agent_file: str, policy_file: str,
                required_policy_functions: tuple = ()):
    """
    Executes a genome and returns (ArchiveAgent class, policy module).

    Raises on a genome whose mutated code does not import or does not define what
    the episode runner needs, so a broken mutation fails once, before any episode
    process is spawned, instead of once per episode.
    """
    genome = read_genome(json_path)

    policy = types.ModuleType("genome_policy")
    exec(compile(source_of(genome, policy_file), policy_file, "exec"), policy.__dict__)
    for required in required_policy_functions:
        if not callable(getattr(policy, required, None)):
            raise ValueError(f"{json_path}: {policy_file} no longer defines {required}().")

    # Files added by a mutation are registered in sys.modules under their
    # basename, so the agent file can `import <name>` them.
    for key, entry in genome.get("files", {}).items():
        if key in (agent_file, policy_file):
            continue
        module_name = os.path.splitext(os.path.basename(key))[0]
        extra_module = types.ModuleType(module_name)
        exec(compile(entry["content"], key, "exec"), extra_module.__dict__)
        sys.modules[module_name] = extra_module

    namespace = {}
    exec(compile(source_of(genome, agent_file), agent_file, "exec"), namespace)
    if "ArchiveAgent" not in namespace:
        raise ValueError(f"{json_path}: {agent_file} no longer defines the ArchiveAgent class.")

    return namespace["ArchiveAgent"], policy
