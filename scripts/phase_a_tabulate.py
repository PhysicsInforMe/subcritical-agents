#!/usr/bin/env python
"""Phase A, LLM part: tabulate the policy of one cell for every formulation
-> results/phase_a/<cell_id>/gaps.json (design notes §4.3, §5, §14.2).

Usage: python scripts/phase_a_tabulate.py configs/phase_a/<cell>.yaml [...]
       python scripts/phase_a_tabulate.py --all configs/phase_a --model qwen0.6b
Cells already written are skipped unless --force. Progress is checkpointed
after every formulation in gaps_partial.json, so a killed run resumes
where it stopped (the 2026-09-09 run was killed by memory pressure on the
16 GB laptop). The prompt NLL (covariate of §14.1 B15) is computed only for
k_mem = 0 cells: it is a property of the formulation and the memory block
only adds a fixed sentence, and it doubles the number of forward passes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import build_manifest, write_json  # noqa: E402
from src.llm import LogitReader  # noqa: E402
from src.llm.phase_a import tabulate_cell  # noqa: E402
from src.llm.tabulate import LABELS, set_labels  # noqa: E402


def run_cell(cfg: dict, reader: LogitReader, out_dir: Path) -> None:
    forms = yaml.safe_load(open(cfg["formulations"], encoding="utf-8"))
    formats = forms["formats"] if cfg["channel"] == "list" else [forms["formats"][0]]
    cell_dir = out_dir / cfg["cell_id"]
    partial = cell_dir / "gaps_partial.json"
    tables = json.load(open(partial, encoding="utf-8")) if partial.exists() else []
    done = {(t["wording"], t["format"]) for t in tables}
    t0 = time.time()
    for wi, wording in enumerate(forms["wordings"]):
        for fmt in formats:
            if (wi, fmt) in done:
                continue
            tab = tabulate_cell(reader, wording, cfg["channel"], cfg["exposure"], cfg["k_mem"], fmt,
                                cfg["max_orderings"], seed=0, with_nll=(cfg["k_mem"] == 0))
            tab.update({"wording": wi, "format": fmt})
            tables.append(tab)
            write_json(partial, tables)
            print(f"  {cfg['cell_id']} w{wi} {fmt}: {tab['n_passes']} passes, {time.time() - t0:.0f} s", flush=True)
    payload = {"manifest": build_manifest(cfg, [0], {"revision": reader.resolved_revision,
                                                     "variant_ids": reader.variant_ids, "dtype": reader.dtype,
                                                     "labels": list(LABELS), "seconds": time.time() - t0}),
               "cell_id": cfg["cell_id"], "tables": tables}
    write_json(cell_dir / "gaps.json", payload)
    partial.unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--all", default=None, help="directory of cell configs")
    ap.add_argument("--model", default=None, help="only cells of this model_key")
    ap.add_argument("--out", default="results/phase_a")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--dtype", default=None, help="float32 or bfloat16; overrides the dtype of the cell config (revision R4-04: the config carries it)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    paths = [Path(p) for p in a.configs]
    if a.all:
        paths += sorted(Path(a.all).glob("*.yaml"))
    cfgs = [yaml.safe_load(open(p, encoding="utf-8")) for p in paths]
    if a.model:
        cfgs = [c for c in cfgs if c["model_key"] == a.model]
    out_dir = Path(a.out)
    readers: dict[tuple, LogitReader] = {}
    for cfg in cfgs:
        target = out_dir / cfg["cell_id"] / "gaps.json"
        if target.exists() and not a.force:
            print(f"skip {cfg['cell_id']} (exists)")
            continue
        labels = tuple(str(x) for x in cfg.get("labels", ["A", "B"]))   # label tokens of the cell, default A/B (2026-09-30)
        set_labels(*labels)   # before the reader is built: it caches the token variants of the labels
        key = (cfg["model"], labels)
        if key not in readers:
            readers = {key: LogitReader(cfg["model"], dtype=a.dtype or cfg.get("dtype", "float32"), revision=cfg.get("revision"))}   # one model in memory at a time; dtype and HF revision pinned in the config (R4-04, R4-05)
        print(f"cell {cfg['cell_id']} labels {labels}", flush=True)
        run_cell(cfg, readers[key], out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
