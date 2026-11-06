"""Typed numerical exports: JSON metadata plus NPZ arrays, never pickle."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


def _fingerprint(press):
    digest = hashlib.sha256()
    for name in (
        "tracking_data",
        "actions_data",
        "player_possessions_data",
        "possessions_data",
        "events_data",
    ):
        table = getattr(press, name)
        digest.update(name.encode())
        if table is not None:
            digest.update(json.dumps(list(table.columns)).encode())
            digest.update(pd.util.hash_pandas_object(table, index=False).values.tobytes())
    digest.update(json.dumps(press.get_match_info(), sort_keys=True).encode())
    return digest.hexdigest()


def _model_fingerprints(press):
    from PRESS.model import (
        ActionSelectionModel, BallDriveModel, PassRiskModel,
        PassSelectionModel, ResidualRiskModel,
    )

    classes = {
        "ball_drive_model": BallDriveModel, "pass_risk_model": PassRiskModel,
        "pass_selection_model": PassSelectionModel,
        "action_selection_model": ActionSelectionModel,
        "residual_risk_model": ResidualRiskModel,
    }
    result = {}
    for name, cls in classes.items():
        model = getattr(press, name) or cls.pretrained()
        text = json.dumps(model.to_dict(), sort_keys=True, allow_nan=False)
        result[name] = hashlib.sha256(text.encode()).hexdigest()
    return result


def _encode(value, arrays):
    if isinstance(value, dict):
        return {
            "kind": "mapping",
            "items": [[_encode(k, arrays), _encode(v, arrays)] for k, v in value.items()],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(v, arrays) for v in value],
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    # NumPy/JAX arrays and nonfinite/numpy scalars retain their dtype and shape.
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Result exports do not support object arrays or arbitrary Python objects.")
    key = f"array_{len(arrays)}"
    arrays[key] = array
    return {"kind": "array", "key": key, "scalar": array.ndim == 0}


def _decode(value, arrays):
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind == "mapping":
        return {_decode(k, arrays): _decode(v, arrays) for k, v in value["items"]}
    if kind in ("tuple", "list"):
        items = [_decode(v, arrays) for v in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "array":
        array = arrays[value["key"]].copy()
        return array.item() if value["scalar"] else array
    raise ValueError("Unknown result-export value type.")


def save_results(press, path, *, overwrite=False):
    """Write a versioned export without modifying the live cache."""
    path = Path(path)
    manifest_path, arrays_path = path / "metadata.json", path / "arrays.npz"
    if path.exists() and not path.is_dir():
        raise ValueError("The result export path must be a directory.")
    if not overwrite and (manifest_path.exists() or arrays_path.exists()):
        raise FileExistsError(f"Results already exist in {path}; use overwrite=True.")
    arrays = {}
    payload = _encode(press.get_results(), arrays)
    manifest = {
        "schema_version": 2, "format": "PRESS-results", "match": press.get_match_info(),
        "input_sha256": _fingerprint(press), "models": _model_fingerprints(press),
        "arrays_file": "arrays.npz", "results": payload,
    }
    path.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path, suffix=".npz", delete=False) as tmp:
        array_tmp = Path(tmp.name)
    metadata_tmp = None
    try:
        np.savez_compressed(array_tmp, **arrays)
        manifest["arrays_sha256"] = hashlib.sha256(array_tmp.read_bytes()).hexdigest()
        with NamedTemporaryFile(dir=path, mode="w", suffix=".json", delete=False) as tmp:
            metadata_tmp = Path(tmp.name)
            json.dump(manifest, tmp, indent=2, allow_nan=False)
            tmp.write("\n")
        os.replace(array_tmp, arrays_path)
        os.replace(metadata_tmp, manifest_path)
    finally:
        array_tmp.unlink(missing_ok=True)
        if metadata_tmp is not None:
            metadata_tmp.unlink(missing_ok=True)
    return path


def load_results(press, path, *, overwrite=False):
    """Validate the complete saved export before updating any cached result."""
    path = Path(path)
    manifest = json.loads((path / "metadata.json").read_text())
    if manifest.get("schema_version") != 2 or manifest.get("format") != "PRESS-results":
        raise ValueError("Unsupported PRESS result export.")
    if manifest.get("arrays_file") != "arrays.npz":
        raise ValueError("Invalid result-array filename.")
    if manifest.get("input_sha256") != _fingerprint(press):
        raise ValueError("Saved results do not match this match's tracking, annotations or metadata.")
    if manifest.get("models") != _model_fingerprints(press):
        raise ValueError("Saved results use different fitted model settings.")
    source = path / "arrays.npz"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest.get("arrays_sha256"):
        raise ValueError("Result arrays checksum mismatch.")
    with np.load(source, allow_pickle=False) as arrays:
        restored = _decode(manifest["results"], arrays)
    if not isinstance(restored, dict) or set(restored) != set(press.results):
        raise ValueError("Saved component names do not match this PRESS version.")
    for component, frames in restored.items():
        if not isinstance(frames, dict):
            raise ValueError("Malformed component results.")
        for frame, result in frames.items():
            if (
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or not isinstance(result, dict)
            ):
                raise ValueError("Malformed frame result.")
            press.get_frame(frame)
            if not overwrite and frame in press.results[component]:
                raise ValueError(f"A {component} result already exists at frame {frame}; use overwrite=True.")
    for component, frames in restored.items():
        press.results[component].update(frames)
    return press.get_results()
