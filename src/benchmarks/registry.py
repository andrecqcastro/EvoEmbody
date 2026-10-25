"""
Looks up a benchmark package by name. The evolution loop and the reporting code
only talk to the active benchmark through the module returned here.
"""
import importlib

_BENCHMARKS = {
    "alfworld": "src.benchmarks.alfworld",
}


def get_benchmark(name: str):
    if name not in _BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {sorted(_BENCHMARKS)}.")
    return importlib.import_module(_BENCHMARKS[name])
