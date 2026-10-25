# Sandbox for executing LLM-authored genome code (episode playing, error mining
# and the Implementation agent). See src/utils/sandbox.py.
#
# Build from the repository root:
#   docker build -f docker/genome_sandbox.Dockerfile -t alfworld-genome-sandbox:latest .
FROM python:3.10-slim

# jericho (pulled in by alfworld) compiles its interpreter at install time.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The project and alfworld_data are bind-mounted at run time at their host
# paths, so code changes never require a rebuild.
ENTRYPOINT ["python", "-m"]
