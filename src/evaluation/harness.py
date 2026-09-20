"""Experiment runner: multiple seeds, aggregated metrics, and a reproducibility manifest.

Every experiment goes through `run_experiment`, which guarantees four things the
project previously lacked (docs/00_INVESTIGATION_REPORT.md, defect B10):

  * results come from several seeds and are reported as mean +/- std, never from a
    single lucky run;
  * each run records the code revision, the seeds and the environment, so a figure
    can be traced back to what produced it;
  * each run publishes an EVIDENCE.md stating what was measured, derived, external,
    simulated and assumed;
  * the provenance guard runs before anything is written, so a synthetic result can
    never be filed as a demonstrated one.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .provenance import Evidence, check_conclusion

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def git_revision(repo: Path | None = None) -> str:
    """Short commit hash, or a clear marker when it cannot be determined."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo or Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=10, check=True,
        )
        rev = out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "UNKNOWN (not a git repository)"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo or Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        dirty = ""
    return f"{rev}-dirty" if dirty else rev


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hash a data file so a result can name the exact bytes it was computed from."""
    h = sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def summarise_across_seeds(per_seed: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """Aggregate per-seed metrics to mean and std. A single seed proves nothing.

    Only NUMERIC columns are aggregated. An experiment may legitimately carry
    descriptive string columns (a protocol label, a variant name) that are not part
    of the index; averaging those is meaningless, so they are dropped from the
    summary rather than raising.
    """
    metrics = [c for c in per_seed.columns
               if c not in index_cols + ["seed"]
               and pd.api.types.is_numeric_dtype(per_seed[c])]
    if not metrics:
        raise ValueError(
            f"No numeric metric columns to summarise. Columns: {list(per_seed.columns)}; "
            f"index_cols={index_cols}."
        )
    grouped = per_seed.groupby(index_cols)[metrics]
    out = grouped.agg(["mean", "std", "count"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.reset_index()


def run_experiment(
    exp_id: str,
    description: str,
    fn: Callable[[int], pd.DataFrame],
    evidence: Evidence,
    out_dir: str | Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
    index_cols: list[str] | None = None,
    config: dict | None = None,
    data_files: list[str | Path] | None = None,
    verbose: bool = True,
) -> dict:
    """Run `fn(seed)` once per seed, aggregate, and write the full record.

    `fn` returns a tidy DataFrame of metrics for that seed (one row per model or
    per condition). `index_cols` names the columns that identify a row across seeds.
    """
    index_cols = index_cols or ["model"]
    check_conclusion(evidence.provenance, evidence.conclusion_type)

    out = Path(out_dir) / exp_id
    out.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    frames = []
    for seed in seeds:
        log(f"[{exp_id}] seed {seed} ...")
        frame = fn(seed).copy()
        frame["seed"] = seed
        frames.append(frame)
    per_seed = pd.concat(frames, ignore_index=True)
    summary = summarise_across_seeds(per_seed, index_cols)

    per_seed.to_csv(out / "metrics_per_seed.csv", index=False)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    (out / "EVIDENCE.md").write_text(evidence.to_markdown())

    manifest = {
        "experiment": exp_id,
        "description": description,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "seeds": list(seeds),
        "n_rows_per_seed": int(len(per_seed) / max(len(seeds), 1)),
        "provenance": evidence.provenance.value,
        "conclusion_type": evidence.conclusion_type.value,
        "split_protocol": evidence.split_protocol,
        "n": evidence.n,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": _jsonable(config or {}),
        "data_files": {str(p): file_sha256(p) for p in (data_files or [])},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    log(f"[{exp_id}] wrote {out}")
    return {"per_seed": per_seed, "summary": summary, "manifest": manifest, "out_dir": out}


def _jsonable(obj):
    """Best-effort conversion so a config containing dataclasses can be recorded."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
