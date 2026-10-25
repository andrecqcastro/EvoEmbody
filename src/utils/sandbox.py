"""
Runs a Python module inside the genome-sandbox Docker container.

Only the steps that execute LLM-authored code go through the sandbox: episode
playing and error mining (both exec() a mutated agent and policy from a genome
JSON) and the Implementation agent (whose output is that genome JSON). The vLLM
servers and the evolution driver stay on the host. Enabled with
GENOME_SANDBOX=on once docker/genome_sandbox.Dockerfile has been built.

Network isolation is best-effort: the container reaches the vLLM servers on the
host through the host-gateway alias, and Docker's default bridge network does
not block other egress. Restricting it further needs a host firewall rule.
"""
import os
import subprocess

IMAGE = os.getenv("SANDBOX_IMAGE", "alfworld-genome-sandbox:latest")


def sandbox_available() -> bool:
    """Whether the sandbox image has been built."""
    try:
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def sandbox_enabled() -> bool:
    """True when GENOME_SANDBOX=on and the image exists. Warns and returns False otherwise."""
    if os.getenv("GENOME_SANDBOX", "off").strip().lower() not in ("on", "1", "true"):
        return False
    if not sandbox_available():
        print(f"  [sandbox] GENOME_SANDBOX=on but image {IMAGE!r} was not found. Build it with "
              f"docker/genome_sandbox.Dockerfile. Running this step unsandboxed.")
        return False
    return True


def rewrite_localhost_for_container(value: str) -> str:
    """Points 127.0.0.1/localhost URLs at the host, as seen from inside the container."""
    return value.replace("127.0.0.1", "host.docker.internal").replace(
        "localhost", "host.docker.internal")


def env_for_container() -> dict:
    """
    The current environment with every 127.0.0.1/localhost URL rewritten. The
    whole environment is forwarded so that sampling settings read by
    src/config/llm_config.py are identical inside and outside the sandbox.
    """
    return {key: rewrite_localhost_for_container(value) if
           ("127.0.0.1" in value or "localhost" in value) else value
           for key, value in os.environ.items()}


def run_module(project_root: str, module: str, args: list, writable_dirs: list,
               timeout: int = None) -> dict:
    """
    Runs `python -m <module> <args...>` inside the sandbox container.

    The project is mounted read-only at its host path, so absolute paths stored
    in archive.json stay valid on both sides. Each directory in writable_dirs
    (which must be under project_root) is mounted read-write on top of it.

    Returns {"returncode", "stdout", "stderr"}.
    """
    project_root = os.path.abspath(project_root)
    cmd = [
        "docker", "run", "--rm",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        # Run as the host user so files written to the bind mounts keep its ownership.
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--add-host", "host.docker.internal:host-gateway",
        "--tmpfs", "/tmp",
        "--workdir", project_root,
        "-v", f"{project_root}:{project_root}:ro",
    ]
    for host_dir in writable_dirs:
        host_dir = os.path.abspath(host_dir)
        if os.path.commonpath([host_dir, project_root]) != project_root:
            raise ValueError(f"{host_dir} is not under the sandboxed project root {project_root}")
        os.makedirs(host_dir, exist_ok=True)
        cmd += ["-v", f"{host_dir}:{host_dir}:rw"]
    for key, value in env_for_container().items():
        cmd += ["-e", f"{key}={value}"]
    # Host cache and temp paths are not mounted in the container.
    cmd += ["-e", "HOME=/tmp", "-e", "TMPDIR=/tmp"]
    cmd += [IMAGE, module, *args]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
