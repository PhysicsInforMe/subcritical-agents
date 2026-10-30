#!/usr/bin/env python
"""Generate one YAML per Phase A cell at the poor channel (design notes §5, §14.2).

Grid (tabulated regime, T analytic from the logits, no designed field):
    model     Qwen3-0.6B, Qwen3-1.7B
    channel   list (peer states shown), count (only the counts, Cho's Count)
    k_mem     0, 3
    exposure  2, 4, 7 (all N-1 for N = 8; the "8" of §5 is impossible at N = 8)
    topology  all-to-all N = 8; lattice L = 6 (exposure 4, k_mem 0) as control
Cell id: A_<model>_<channel>_k<k>_e<e>_Tan_f0_<topo>_N<N>
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

MODELS = {"qwen0.6b": "Qwen/Qwen3-0.6B", "qwen1.7b": "Qwen/Qwen3-1.7B"}
COMMON = {
    "phase": "A", "f_spec": 0.0, "temperatures": [0.5, 1.0, 2.0, 4.0],
    "chi_scan": {"T_min": 0.25, "T_max": 64.0, "n_T": 13},
    "max_orderings": 8, "seeds": list(range(1, 31)), "n_rounds": 20, "n_boot": 300,
    "formulations": "configs/phase_a_formulations.yaml",
    "stubborn_z": [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5],
}


def main(out_dir: str = "configs/phase_a") -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for mkey, mname in MODELS.items():
        for channel in ("list", "count"):
            for k_mem in (0, 3):
                for exposure in (2, 4, 7):
                    cells.append(dict(model_key=mkey, model=mname, channel=channel, k_mem=k_mem,
                                      exposure=exposure, topology="all", N=8))
        cells.append(dict(model_key=mkey, model=mname, channel="list", k_mem=0, exposure=4,
                          topology="lat6", N=36))
    for c in cells:
        cid = f"A_{c['model_key']}_{c['channel']}_k{c['k_mem']}_e{c['exposure']}_Tan_f0_{c['topology']}_N{c['N']}"
        cfg = {"cell_id": cid, **c, **COMMON}
        with (out / f"{cid}.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    print(f"{len(cells)} cell configs written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
