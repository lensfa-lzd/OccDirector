"""CARLA host pool loading shared by the data engines.

Hosts live in a gitignored JSON file (see hosts.example.json) because they
describe private infrastructure::

    [{"ip": "10.0.0.1", "workers": 8, "base_port": [carla_start, tm_start]}]
"""
import json
import os

_DEFAULT_EXAMPLE = os.path.join(os.path.dirname(__file__), 'hosts.example.json')


def load_host_resources(path: str) -> list:
    """Load [{"ip", "workers", "base_port"}] entries from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        resources = json.load(f)
    if not resources:
        raise ValueError(f"no host resources defined in {path}")
    return resources


def default_hosts_path(module_file: str) -> str:
    """hosts.json next to the calling module, falling back to the shared example."""
    candidate = os.path.join(os.path.dirname(module_file), 'hosts.json')
    return candidate if os.path.exists(candidate) else _DEFAULT_EXAMPLE
