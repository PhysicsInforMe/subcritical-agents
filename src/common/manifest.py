"""Run manifests.

Every run writes, next to its raw results, a manifest with the full config,
the seeds, the git commit of the code, a timestamp and a description of the
hardware and library versions (design notes §14.2). De Nobili et al. report that
identical seeds give slightly different logits on different GPU
architectures (§10.2, hardware reproducibility), so the hardware block is
not decorative: it is what lets a later reader explain a small shift.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def git_commit_hash(short: bool = False) -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out[:8] if short else out


def git_tree_state() -> dict[str, Any]:
    """Whether the working tree differed from HEAD when the run started (revision R4-06):
    ``dirty`` is True if ``git status --porcelain`` lists anything, and ``diff_sha1`` is
    the SHA-1 of ``git diff HEAD`` so that an uncommitted change can be recognised later.
    Since 2026-09-30 (commit 1a0ae99) the diff is read as bytes and decoded as UTF-8 with
    replacement; hashes written before that date were computed on the diff read in text
    mode (system encoding, CRLF normalised to LF) and are not comparable with the new ones."""
    try:
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, check=True).stdout.decode("utf-8", "replace")   # bytes: the diff may hold any encoding (2026-09-30)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"dirty": None, "diff_sha1": None}
    import hashlib
    return {"dirty": bool(status.strip()), "diff_sha1": hashlib.sha1(diff.encode("utf-8", "replace")).hexdigest() if diff.strip() else None}


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def build_manifest(config: dict[str, Any], seeds: list[int],
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the manifest block required by scripts/check_phase_0.py."""
    manifest: dict[str, Any] = {
        "config": config,
        "seeds": list(int(s) for s in seeds),
        "commit": git_commit_hash(),
        "tree": git_tree_state(),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "hardware": _hardware(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as UTF-8 JSON with stable key order, creating dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return path
