#!/usr/bin/env python
"""Reproduce every analysis of the paper from the committed gap tables (revision R4-02).

The chain from the raw ``gaps.json`` (and the ``measures.json`` of the dynamics) to
``paper/numbers.json``, ``paper/tables.md``, ``paper/arxiv/tab_*.tex`` and
``figures/paper/`` is the ordered list STEPS below, each with the exact arguments
that produced the committed file (taken from the ``config`` block of its manifest).
Model runs (tabulation, generation check, De Nobili literal prompt) and the long
dynamics runs (``phase_a_dynamics``, ``phase_a_lattice``, ``phase_a_mf_dynamics_check``,
``check_stubborn_placement``) are not repeated here: their outputs are the inputs.

    python scripts/reproduce.py            # run every step, overwrite the result files
    python scripts/reproduce.py --check    # run every step into a temporary copy and
                                           # report, per file, whether the regenerated
                                           # JSON equals the committed one outside the
                                           # manifest block (log: results/reproduce_check.log)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STEPS: list[tuple[list[str], list[str]]] = [
    # (command, files produced)
    (["scripts/phase_a_aggregate.py"], ["results/phase_a/aggregate.json"]),
    (["scripts/phase_a_covariates.py"], ["results/phase_a/covariates.json"]),
    (["scripts/phase_a_orderings_check.py", "--exposures", "7,10,30", "--models", "qwen0.6b,qwen1.7b", "--n-boot", "300"], ["results/phase_a/orderings_check.json"]),
    (["scripts/phase_a_orderings_subset.py", "--exposures", "10,30", "--models", "qwen0.6b,qwen1.7b", "--n-subsets", "200", "--subset-size", "8"], ["results/phase_a/orderings_subset.json"]),
    (["scripts/phase_a_meanfield_stability.py", "--models", "qwen0.6b,qwen1.7b", "--T", "1.0"], ["results/phase_a/meanfield_stability.json"]),
    (["scripts/phase_a_format_check.py", "--models", "qwen0.6b,qwen1.7b,smollm1.7b", "--e-ref", "7"], ["results/phase_a/format_check.json"]),
    (["scripts/phase_a_exposure_shape.py", "--models", "qwen0.6b,qwen1.7b", "--e-ref", "7"], ["results/phase_a/exposure_shape.json"]),
    (["scripts/phase_a_memory_curve.py", "--models", "qwen0.6b,qwen1.7b", "--exposure", "4"], ["results/phase_a/memory_curve.json"]),
    (["scripts/phase_a_dF0_classes.py", "--models", "qwen0.6b,qwen1.7b", "--n-boot", "300", "--out", "results/phase_a/dF0_classes.json"], ["results/phase_a/dF0_classes.json"]),
    (["scripts/phase_a_dF0_classes.py", "--models", "qwen1.7b", "--n-boot", "300", "--results", "results/phase_a_fp32check", "--out", "results/phase_a/dF0_classes_fp32.json"], ["results/phase_a/dF0_classes_fp32.json"]),
    (["scripts/phase_a_h10_test.py", "--models", "qwen0.6b,qwen1.7b,smollm1.7b"], ["results/phase_a/h10_test.json"]),
    (["scripts/phase_a_second_family.py", "--models", "smollm1.7b,qwen0.6b,qwen1.7b", "--e-ref", "7"], ["results/phase_a/second_family.json"]),
    (["scripts/phase_a_saturation_fit.py", "--models", "qwen0.6b,qwen1.7b"], ["results/phase_a/saturation_fit.json"]),
    (["scripts/phase_a_fp32_recheck.py", "--model", "qwen1.7b", "--exposures", "10,30", "--n-boot", "300", "--bf16", "results/phase_a", "--fp32", "results/phase_a_fp32check"], ["results/phase_a/fp32_recheck.json"]),
    (["scripts/review_r1_checks.py"], ["results/phase_a/review_r1.json"]),
    (["scripts/review_r2_checks.py"], ["results/phase_a/review_r2.json", "paper/appendix_prompts.md"]),
    (["scripts/review_r2_runs_analysis.py"], ["results/phase_a/review_r2_runs.json"]),
    (["scripts/review_r3_checks.py"], ["results/phase_a/review_r3.json"]),
    (["scripts/review_r3_newwordings.py"], ["results/phase_a/review_r3_newwordings.json"]),
    (["scripts/review_newwordings_ba.py"], ["results/phase_a/review_newwordings_ba.json"]),
    (["scripts/review_labels_xy.py"], ["results/phase_a/review_labels_xy.json"]),
    (["scripts/phase_a_F0_decomposition.py", "--models", "qwen0.6b,qwen1.7b", "--exposure", "4", "--noise-sd", "qwen0.6b=0.12,qwen1.7b=0.37", "--n-draws", "400", "--seed", "0"], ["results/phase_a/F0_decomposition_e4.json"]),
    (["scripts/paper_numbers.py"], ["paper/numbers.json", "paper/numbers.md"]),
    (["scripts/paper_tables.py"], ["paper/tables.md", "paper/arxiv/tab_exposure.tex", "paper/arxiv/tab_formulations.tex", "paper/arxiv/tab_decomposition.tex"]),
    (["scripts/paper_figures.py"], []),
]


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k != "manifest"}
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated script names to run (default: all)")
    a = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    only = set(a.only.split(",")) if a.only else None
    if a.check:
        tmp = Path(tempfile.mkdtemp(prefix="reproduce_"))
        for d in ("scripts", "src", "configs", "results", "paper", "tests"):
            shutil.copytree(root / d, tmp / d, ignore=shutil.ignore_patterns("*.png"))
        work = tmp
        log = open(root / "results" / "reproduce_check.log", "w", encoding="utf-8")
        log.write(f"# reproduce --check, HEAD {subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=root).stdout.strip()}\n")
    else:
        work = root; log = None
    env = dict(os.environ); env["PYTHONPATH"] = str(work)
    ok_all = True
    for cmd, files in STEPS:
        name = Path(cmd[0]).stem
        if only and name not in only:
            continue
        r = subprocess.run([sys.executable, *cmd], cwd=work, env=env, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
        line = f"{' '.join(cmd)}: {status}"
        if a.check and r.returncode == 0:
            for f in files:
                if f.endswith(".json"):
                    new = _strip(json.load(open(work / f, encoding="utf-8")))
                    old = _strip(json.load(open(root / f, encoding="utf-8"))) if (root / f).exists() else None
                    same = new == old
                    ok_all &= same
                    line += f" | {f}: {'identical outside the manifest' if same else 'DIFFERENT'}"
                elif f.endswith((".md", ".tex")):
                    same = (work / f).read_text(encoding="utf-8") == (root / f).read_text(encoding="utf-8")
                    ok_all &= same
                    line += f" | {f}: {'identical' if same else 'DIFFERENT'}"
        elif r.returncode != 0:
            ok_all = False; line += "\n" + r.stderr[-800:]
        print(line, flush=True)
        if log:
            log.write(line + "\n")
    if log:
        log.write("# all identical\n" if ok_all else "# differences found\n"); log.close()
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
