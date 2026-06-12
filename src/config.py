"""Central configuration loader.

Resolves all paths relative to the project root so the project runs on any
machine with no hardcoded absolute paths. Loads ``config.yaml`` once and
exposes it as a plain dict plus a few convenience helpers.
"""
import os
import yaml

# Project root = parent of this file's directory (src/ -> project root).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def load_config(path=CONFIG_PATH):
    """Load config.yaml and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str):
    """Resolve a (possibly relative) path against the project root."""
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(PROJECT_ROOT, path_str)


def get_paths(cfg=None):
    """Return resolved (data_file, models_dir, logs_dir, images_dir, results_dir)
    and create the output directories if missing."""
    cfg = cfg or load_config()
    p = cfg["paths"]
    data_file = resolve(p["data_file"])
    models_dir = resolve(p["models_dir"])
    logs_dir = resolve(p["logs_dir"])
    images_dir = resolve(p["images_dir"])
    results_dir = resolve(p["results_dir"])
    for d in (models_dir, logs_dir, images_dir, results_dir):
        os.makedirs(d, exist_ok=True)
    return data_file, models_dir, logs_dir, images_dir, results_dir


# Loaded once on import for convenience.
CONFIG = load_config()
