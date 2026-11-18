"""Small file helpers for the procedural review; no model or stage dispatch."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parent
ACTION_KEYS = ["match_id", "action_id", "frame_start"]
PARENT_KEYS = ["match_id", "possession_id"]


def read_table(path):
    """Identifiers have one dtype in every task, independently of parquet."""
    table = pd.read_parquet(path)
    if "match_id" in table:
        table["match_id"] = table.match_id.astype(str)
    return table


def new_folder(path):
    """A completed output is never silently replaced; choose a fresh run root."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, values):
    """JSON stores settings and summaries; arrays/tables are saved separately."""
    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [convert(v) for v in value]
        if isinstance(value, np.generic):
            return convert(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(values), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_inventory(folder):
    """Record relative names and hashes, without exposing local source paths."""
    folder = Path(folder)
    files = {str(p.relative_to(folder)): file_hash(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "inventory.json"}
    write_json(folder / "inventory.json", files)
