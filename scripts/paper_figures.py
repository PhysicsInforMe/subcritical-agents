#!/usr/bin/env python
"""Figures of the full paper and of the lay version, generated from the result
files (never from transcribed numbers). Output: figures/paper/*.png.

Conventions (dataviz conventions, validated palette on the light surface): one axis
per chart, categorical hues in fixed order (0.6B blue, 1.7B orange, SmolLM2
aqua; formats blue, orange, aqua, yellow), thin lines, markers of at least 8
px, direct labels, neutral titles; the reading rules go in the captions of
the documents, not in the image.

Sources: results/phase_a/<cell>/gaps.json (curves and jumps recomputed with
the estimators of phase_a_format_check), dF0_classes.json, mf_dynamics_check.json,
memory_curve.json, lattice_control.json, stubborn_placement.json,
meanfield_stability (mf_map) for the lay illustration of the map.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase_a_format_check import j_gap, load_mean_gaps  # noqa: E402
from scripts.phase_a_meanfield_stability import mf_map, policy_from_entries  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"
MODEL_COL = {"qwen0.6b": "#2a78d6", "qwen1.7b": "#eb6834", "smollm1.7b": "#1baf7a"}
MODEL_LAB = {"qwen0.6b": "Qwen3-0.6B", "qwen1.7b": "Qwen3-1.7B", "smollm1.7b": "SmolLM2-1.7B"}
FMT_COL = {"comma": "#2a78d6", "lines": "#eb6834", "semicolon": "#1baf7a", "brackets": "#eda100"}
FMT_LAB = {"comma": "comma", "lines": "lines", "semicolon": "semicolon", "brackets": "brackets"}
R = Path("results/phase_a")
OUT = Path("figures/paper")

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 10, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 8})


def cluster_band(vals: np.ndarray, words: np.ndarray, rng: np.random.Generator = None, n: int = 2000) -> tuple[float, float]:
    """95% t interval on the per-wording means (the wording is the independent unit; revision R3-01, R4-08).
    The percentile cluster bootstrap used in a first version is not drawn any more."""
    from scipy import stats
    m = np.array([vals[words == c].mean() for c in np.unique(words)])
    h = float(stats.t.ppf(0.975, m.size - 1) * m.std(ddof=1) / np.sqrt(m.size)) if m.size > 1 else 0.0
    return float(m.mean() - h), float(m.mean() + h)


def curves(model: str) -> tuple[list[int], dict, dict]:
    """J_gap per (wording, format, e) from the 8-ordering cells, and the mean gap tables."""
    G = load_mean_gaps(R, model)
    es = sorted({e for (_, _, e) in G})
    J = {k: j_gap(g) for k, g in G.items()}
    return es, J, G


def ord32_points(model: str) -> dict[int, float]:
    """Mean J_gap over formulations from the 32-ordering cells (7, 10, 30 peers) where present."""
    out = {}
    for e, n in ((7, 8), (10, 32), (30, 32)):
        p = R / f"A_{model}_list_k0_e{e}_Tan_f0_allo32_N{n}" / "gaps.json"
        if p.exists():
            tabs = json.load(open(p, encoding="utf-8"))["tables"]
            vals = []
            for t in tabs:
                byk: dict[int, list[float]] = {}
                for x in t["entries"]:
                    byk.setdefault(x["k"], []).append(x["gap"])
                vals.append(j_gap({k: float(np.mean(v)) for k, v in byk.items()}))
            out[e] = float(np.mean(vals))
    return out


def fig1_coupling_vs_exposure(rng):
    models = [m for m in ("qwen0.6b", "qwen1.7b", "smollm1.7b") if (R / f"A_{m}_list_k0_e2_Tan_f0_all_N8").exists()]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        es, J, _ = curves(m)
        forms = sorted({(w, f) for (w, f, e) in J})
        fmts = [f for f in FMT_COL if any(ff == f for (_, ff) in forms)]
        for f in fmts:
            keys = [(w, ff) for (w, ff) in forms if ff == f]
            ys = [np.mean([J[(w, f, e)] for (w, _) in keys]) for e in es]
            ax.plot(es, ys, "-o", color=FMT_COL[f], label=FMT_LAB[f], markeredgecolor=SURFACE, markeredgewidth=1)
        allkeys = forms
        lo, hi = [], []
        for e in es:
            v = np.array([J[(w, f, e)] for (w, f) in allkeys]); wd = np.array([w for (w, _) in allkeys])
            a, b = cluster_band(v, wd, rng); lo.append(a); hi.append(b)
        ax.fill_between(es, lo, hi, color=INK2, alpha=0.12, linewidth=0, label="all formats, cluster 95% band")
        pts = ord32_points(m)
        if pts:
            ax.plot(list(pts), list(pts.values()), "s", color=INK, markerfacecolor=SURFACE, markersize=9, label="32 orderings, mean")
        ax.set_xscale("log"); ax.set_xticks(es); ax.set_xticklabels([str(e) for e in es])
        ax.set_title(MODEL_LAB[m], color=INK, loc="left")
        ax.set_xlabel("peers shown at each update (exposure)")
    axes[0].set_ylabel("coupling per peer J̃ (logit per peer)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig1_coupling_vs_exposure.png", dpi=180); plt.close(fig)


def fig2_jump_vs_exposure():
    models = [m for m in ("qwen0.6b", "qwen1.7b", "smollm1.7b") if (R / f"A_{m}_list_k0_e2_Tan_f0_all_N8").exists()]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8))
    axes = np.atleast_1d(axes)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#bcd3f2", "#2a78d6", "#0f3b73"])
    for ax, m in zip(axes, models):
        es, _, G = curves(m)
        forms = sorted({(w, f) for (w, f, e) in G if all((w, f, x) in G for x in es)})
        j7 = {k: G[(k[0], k[1], 7)][7] - G[(k[0], k[1], 7)][-7] for k in forms}
        lo, hi = min(j7.values()), max(j7.values())
        for k in sorted(forms, key=lambda q: j7[q]):
            ys = [G[(k[0], k[1], e)][e] - G[(k[0], k[1], e)][-e] for e in es]
            c = cmap((j7[k] - lo) / (hi - lo + 1e-9))
            ax.plot(es, ys, "-", color=c, linewidth=1.6, alpha=0.9)
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1.0); ax.set_xticks(es); ax.set_xticklabels([str(e) for e in es])
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(MODEL_LAB[m], color=INK, loc="left"); ax.set_xlabel("peers shown (exposure)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(lo, hi)); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02); cb.set_label("jump at 7 peers (logit)", fontsize=8)
    axes[0].set_ylabel("unanimity jump gap(+e) − gap(−e) (logit)")
    fig.tight_layout(); fig.savefig(OUT / "fig2_jump_vs_exposure.png", dpi=180); plt.close(fig)


def fig3_dF0_classes():
    d = json.load(open(R / "dF0_classes.json", encoding="utf-8"))["models"]
    dyn = json.load(open(R / "mf_dynamics_check.json", encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, m in zip(axes[:2], ("qwen0.6b", "qwen1.7b")):
        blocks = d[m]
        # prefer 32-ordering cells where both exist
        by_e: dict[int, dict] = {}
        for key, b in blocks.items():
            e = b["exposure"]
            if e not in by_e or b["n_orderings"] > by_e[e]["n_orderings"]:
                by_e[e] = b
        es = sorted(by_e)
        for i, e in enumerate(es):
            rows = by_e[e]["rows"]
            xs = i + np.linspace(-0.3, 0.3, len(rows))
            for x, r in zip(xs, rows):
                lo, hi = r["ci_orderings"]
                col = {"ordered": "#eb6834", "disordered": "#2a78d6"}.get(r["class"].replace("_exact", ""), INK2)
                ax.plot([x, x], [lo, hi], "-", color=col, linewidth=1.4, alpha=0.8)
                ax.plot(x, r["dF0"], "o", color=col, markersize=6, markeredgecolor=SURFACE)
        ax.axhline(1, color=INK, linewidth=1, linestyle="--")
        ceil = [sum(math.comb(e, n) / 2 ** e * abs(2 * n - e) for n in range(e + 1)) for e in es]  # E0|k|, the combinatorial ceiling of F'(0)
        ax.plot(range(len(es)), ceil, "_", color=INK, markersize=22, markeredgewidth=1.2, linestyle=":", linewidth=0.8, label="ceiling E₀|k| (majority rule)")
        ax.set_xticks(range(len(es))); ax.set_xticklabels([f"{e}\n({by_e[e]['n_orderings']} ord.)" for e in es])
        ax.set_xlabel("peers shown (exposure)"); ax.set_title(MODEL_LAB[m], color=INK, loc="left")
        ax.plot([], [], "o", color="#eb6834", label="ordered (interval above 1)")
        ax.plot([], [], "o", color="#2a78d6", label="disordered (interval below 1)")
        ax.plot([], [], "o", color=INK2, label="undetermined")
    axes[0].set_ylabel("slope of the mean-field map at m = 0, F'(0)")
    axes[0].legend(fontsize=8, loc="upper left")
    ax = axes[2]
    for m in ("qwen0.6b", "qwen1.7b"):
        rows = dyn["models"][m]["rows"]
        ax.errorbar([r["dF0_sym"] for r in rows], [r["abs_m_final"] for r in rows], yerr=[r["abs_m_sd_seeds"] for r in rows],
                    fmt="o", color=MODEL_COL[m], markersize=7, markeredgecolor=SURFACE, elinewidth=1, label=MODEL_LAB[m])
    ax.axvline(1, color=INK, linewidth=1, linestyle="--"); ax.set_xlabel("F'(0) of the table (4 peers)")
    ax.set_ylabel(f"final |m| at N = {dyn['N']} ({dyn['rounds']} rounds, {len(dyn['seeds']) if isinstance(dyn['seeds'], list) else dyn['seeds']} seeds)")
    ax.set_title("dynamics against the criterion", color=INK, loc="left"); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig3_meanfield_classes.png", dpi=180); plt.close(fig)


def fig4_memory():
    d = json.load(open(R / "memory_curve.json", encoding="utf-8"))["models"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for m in ("qwen0.6b", "qwen1.7b"):
        ks = sorted(int(k) for k in d[m]["J_gap_mean"])
        axes[0].plot(ks, [d[m]["J_gap_mean"][str(k)] for k in ks], "-o", color=MODEL_COL[m], label=MODEL_LAB[m], markeredgecolor=SURFACE)
        km = sorted(int(k) for k in d[m]["J_mem_gap"])
        mean = [d[m]["J_mem_gap"][str(k)]["mean"] for k in km]
        # t interval on the per-wording means of J_mem (formulation index i belongs to wording i // 2)
        bands = [cluster_band(np.array(d[m]["J_mem_gap"][str(k)]["by_formulation"]), np.arange(len(d[m]["J_mem_gap"][str(k)]["by_formulation"])) // 2) for k in km]
        lo = [b[0] for b in bands]; hi = [b[1] for b in bands]
        axes[1].plot(km, mean, "-o", color=MODEL_COL[m], label=MODEL_LAB[m], markeredgecolor=SURFACE)
        axes[1].fill_between(km, lo, hi, color=MODEL_COL[m], alpha=0.15, linewidth=0)
    axes[0].set_xlabel("memory length k (own past states shown)"); axes[0].set_ylabel("coupling to the peers J̃ (4 peers)")
    axes[0].set_xticks([0, 1, 2, 3]); axes[0].set_title("coupling to the peers against memory length", color=INK, loc="left")
    axes[1].set_xlabel("memory length k"); axes[1].set_ylabel("persistence per memory state J_mem")
    axes[1].set_xticks([1, 2, 3]); axes[1].set_title("persistence against memory length", color=INK, loc="left")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig4_memory_curve.png", dpi=180); plt.close(fig)


def fig5_controls():
    l = json.load(open(R / "lattice_control.json", encoding="utf-8"))["models"]
    s = json.load(open(R / "stubborn_placement.json", encoding="utf-8"))["models"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for m in ("qwen0.6b", "qwen1.7b"):
        for topo, ls in (("all", "-"), ("lattice", "--")):
            blk = l[m]["size_scan"][topo]
            pts = sorted((v["N"], v["frac_consensus"]) for v in blk.values() if isinstance(v, dict))
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], ls, marker="o", color=MODEL_COL[m],
                         markeredgecolor=SURFACE, label=f"{MODEL_LAB[m]}, {'complete graph' if topo == 'all' else 'lattice'}")
    axes[0].set_xscale("log"); axes[0].set_xticks([8, 16, 36, 64]); axes[0].set_xticklabels(["8", "16", "36", "64"])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_xlabel("number of agents N"); axes[0].set_ylabel("fraction of runs reaching consensus")
    axes[0].set_title("consensus against size at fixed table (4 peers)", color=INK, loc="left"); axes[0].legend(fontsize=7)
    conds = ["all8|contiguous", "all8|random", "lattice6|contiguous", "lattice6|random"]
    labels = ["complete graph\ncontiguous", "complete graph\nrandom", "lattice\ncontiguous", "lattice\nrandom"]
    for i, m in enumerate(("qwen0.6b", "qwen1.7b")):
        x = np.arange(len(conds)) + (i - 0.5) * 0.36
        means = [s[m]["means"][c] for c in conds]
        axes[1].bar(x, means, width=0.34, color=MODEL_COL[m], label=MODEL_LAB[m], edgecolor=SURFACE, linewidth=2)
        for j, c in enumerate(conds):
            ys = [r[c] for r in s[m]["rows"]]
            axes[1].plot(np.full(len(ys), x[j]) + np.linspace(-0.08, 0.08, len(ys)), ys, "o", color=INK, markersize=3, alpha=0.6)
    axes[1].set_xticks(range(len(conds))); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("hysteresis gap under stubborn agents"); axes[1].set_title("hysteresis by placement and topology", color=INK, loc="left")
    axes[1].legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout(); fig.savefig(OUT / "fig5_controls.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def fig6_format(rng):
    """One panel per model: mean J_gap by format with its own cluster band over
    the five wordings, at every exposure; the bands overlap if the format does
    not separate the curves (claims B10)."""
    models = [m for m in ("qwen0.6b", "qwen1.7b") if (R / f"A_{m}_list_k0_e2_Tan_f0_allf2_N8").exists()]
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 3.8))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        es, J, _ = curves(m)
        forms = sorted({(w, f) for (w, f, e) in J})
        for f in [f for f in FMT_COL if any(ff == f for (_, ff) in forms)]:
            keys = [(w, ff) for (w, ff) in forms if ff == f]
            ys, lo, hi = [], [], []
            for e in es:
                v = np.array([J[(w, f, e)] for (w, _) in keys]); wd = np.array([w for (w, _) in keys])
                ys.append(float(v.mean())); a, b = cluster_band(v, wd, rng); lo.append(a); hi.append(b)
            ax.plot(es, ys, "-o", color=FMT_COL[f], label=FMT_LAB[f], markeredgecolor=SURFACE, markeredgewidth=1)
            ax.fill_between(es, lo, hi, color=FMT_COL[f], alpha=0.15, linewidth=0)
        ax.set_xscale("log"); ax.set_xticks(es); ax.set_xticklabels([str(e) for e in es])
        ax.set_title(MODEL_LAB[m], color=INK, loc="left"); ax.set_xlabel("peers shown at each update (exposure)")
    axes[0].set_ylabel("coupling per peer J̃ (mean over wordings, 95% band)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig6_format.png", dpi=180); plt.close(fig)


def figL1_pipeline():
    """Lay illustration: from the prompt to the table to the population."""
    fig, ax = plt.subplots(figsize=(10, 2.6)); ax.axis("off")
    boxes = [(0.02, "prompt\n\"Player 1 sees A, A, B, A.\nAnswer with the state…\""),
             (0.27, "two scores from the model\nfor the labels A and B\n(read once, no text written)"),
             (0.52, "table: preference for A\nfor every configuration\nof the peers seen"),
             (0.77, "population of agents\nupdated from the table\nfor hundreds of rounds")]
    for x, txt in boxes:
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 0.15), 0.21, 0.7, boxstyle="round,pad=0.02", fc="#eef3fb", ec="#2a78d6", lw=1.5))
        ax.text(x + 0.105, 0.5, txt, ha="center", va="center", fontsize=9, color=INK)
    for x in (0.235, 0.485, 0.735):
        ax.annotate("", xy=(x + 0.035, 0.5), xytext=(x, 0.5), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))
    fig.savefig(OUT / "figL1_pipeline.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def figL3_configurations():
    """Lay illustration: the sixteen peer lists at 4 peers, two label scores per list, one table row each."""
    import itertools
    lists = list(itertools.product("AB", repeat=4))
    fig, ax = plt.subplots(figsize=(10, 5.2)); ax.axis("off")
    ax.text(0.02, 0.97, "1. the sixteen lists of four peers", fontsize=10, color=INK, weight="bold", va="top")
    for i, L in enumerate(lists):
        col = i // 8; row = i % 8
        x = 0.02 + col * 0.13; y = 0.88 - row * 0.095
        nA = L.count("A")
        ax.text(x, y, " ".join(L), fontsize=10, family="monospace", color=INK, va="center")
        ax.text(x + 0.085, y, f"({nA} A)", fontsize=8, color=INK2, va="center")
    # arrow to the reading step
    ax.annotate("", xy=(0.36, 0.55), xytext=(0.29, 0.55), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))
    ax.add_patch(matplotlib.patches.FancyBboxPatch((0.37, 0.30), 0.24, 0.5, boxstyle="round,pad=0.02", fc="#eef3fb", ec="#2a78d6", lw=1.5))
    ax.text(0.49, 0.72, "2. one prompt per list", fontsize=10, color=INK, weight="bold", ha="center")
    ax.text(0.49, 0.55, "\"Player 1 sees 4 other players\nand their states are: A, A, B, A.\nAnswer with the state Player 1\nshould hold now…\"", fontsize=8.5, ha="center", va="center", color=INK, style="italic")
    ax.text(0.49, 0.37, "the model computes the scores of the\nnext word; we read two of them,\nthe score of A and the score of B", fontsize=8.5, ha="center", va="center", color=INK2)
    ax.annotate("", xy=(0.68, 0.55), xytext=(0.62, 0.55), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))
    ax.text(0.84, 0.97, "3. the table (one row per list)", fontsize=10, color=INK, weight="bold", va="top", ha="center")
    rows = [("A A A A", "+"), ("A A A B", "+"), ("A A B A", "+"), ("…", ""), ("A A B B", "≈ 0"), ("…", ""), ("B B B A", "−"), ("B B B B", "−")]
    ax.text(0.72, 0.88, "list seen", fontsize=8.5, color=INK2); ax.text(0.90, 0.88, "preference\nfor A", fontsize=8.5, color=INK2, ha="center", va="center")
    for k, (L, pref) in enumerate(rows):
        y = 0.80 - k * 0.075
        ax.text(0.72, y, L, fontsize=9.5, family="monospace", color=INK, va="center")
        ax.text(0.90, y, pref, fontsize=10, color="#eb6834" if pref == "+" else ("#2a78d6" if pref == "−" else INK2), ha="center", va="center", weight="bold")
    ax.plot([0.70, 0.98], [0.845, 0.845], color=INK2, lw=0.8)
    ax.text(0.84, 0.16, "+ prefers A, − prefers B, ≈ 0 undecided.\nThe slope of the preference against the\nnumber of A in the list is the coupling J̃.", fontsize=8.5, ha="center", va="center", color=INK2)
    fig.savefig(OUT / "figL3_configurations.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def figL4_topologies():
    """Lay illustration: complete graph of 8 agents against a 6 x 6 lattice with 4 fixed neighbours."""
    import networkx as nx
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax in axes:
        ax.set_aspect("equal"); ax.axis("off")
    G = nx.complete_graph(8); pos = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos, ax=axes[0], edge_color="#c9c8c4", width=1.0)
    nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=["#eb6834"] + ["#2a78d6"] * 7, node_size=260, edgecolors=SURFACE)
    axes[0].set_title("everyone sees everyone (8 agents)\nthe orange agent reads a random sample of e of the others each round", fontsize=9, color=INK, loc="left")
    L = 6; H = nx.grid_2d_graph(L, L, periodic=True); pos2 = {n: (n[0], n[1]) for n in H}
    centre = (2, 3); nb = list(H.neighbors(centre))
    cols = ["#eb6834" if n == centre else ("#eda100" if n in nb else "#2a78d6") for n in H]
    nx.draw_networkx_edges(H, pos2, ax=axes[1], edge_color="#c9c8c4", width=0.8, edgelist=[e for e in H.edges if abs(e[0][0] - e[1][0]) + abs(e[0][1] - e[1][1]) == 1])
    nx.draw_networkx_nodes(H, pos2, ax=axes[1], node_color=cols, node_size=140, edgecolors=SURFACE)
    axes[1].set_title("lattice 6 × 6 (36 agents)\nthe orange agent sees only its four fixed neighbours (yellow)", fontsize=9, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "figL4_topologies.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def figL5_slope():
    """Lay illustration: how J~ is read from a table (preference against #A - #B, slope = J~) for a weak
    and a strong formulation of the 0.6B at 4 peers, and how its uncertainty is measured (ordering bootstrap)."""
    rng = np.random.default_rng(7)
    tabs = {(t["wording"], t["format"]): t for t in json.load(open(R / "A_qwen0.6b_list_k0_e4_Tan_f0_all_N8" / "gaps.json", encoding="utf-8"))["tables"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for key, col, lab in (((0, "comma"), "#2a78d6", "frase debole"), ((4, "lines"), "#eb6834", "frase forte")):
        ents = tabs[key]["entries"]
        ks = np.array([x["k"] for x in ents], float); ys = np.array([x["gap"] / 2 for x in ents])
        ax.plot(ks + rng.normal(0, 0.05, ks.size), ys, "o", color=col, markersize=6, alpha=0.75, markeredgecolor=SURFACE)
        byk = {k: ys[ks == k].mean() for k in np.unique(ks)}
        kk = np.array(sorted(byk)); mm = np.array([byk[k] for k in kk])
        slope, inter = np.polyfit(kk, mm, 1)
        xx = np.linspace(-4, 4, 50); ax.plot(xx, inter + slope * xx, "-", color=col, linewidth=2, label=f"{lab}: pendenza J̃ = {slope:.2f}")
    ax.axhline(0, color=INK2, linewidth=0.8); ax.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_xticklabels(["BBBB\n(0 A)", "3 B, 1 A", "2 e 2", "3 A, 1 B", "AAAA\n(4 A)"])
    ax.set_xlabel("cosa vede l'agente: quanti A meno quanti B nella lista"); ax.set_ylabel("preferenza per A letta nel modello")
    ax.set_title("1. ogni punto è una riga della tabella; la retta che li segue ha pendenza J̃", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=8.5, loc="upper left")
    # panel 2: ordering bootstrap of the slope for the strong formulation
    ax = axes[1]
    ents = tabs[(4, "lines")]["entries"]
    groups = {}
    for x in ents:
        groups.setdefault(x["k"], []).append(x["gap"] / 2)
    slopes = []
    for _ in range(300):
        kk = np.array(sorted(groups), float)
        mm = np.array([np.mean(rng.choice(groups[int(k)], size=len(groups[int(k)]), replace=True)) for k in kk])
        slopes.append(np.polyfit(kk, mm, 1)[0])
    slopes = np.array(slopes); lo, hi = np.percentile(slopes, [2.5, 97.5])
    ax.hist(slopes, bins=25, color="#eb6834", alpha=0.6, edgecolor=SURFACE)
    ax.axvline(lo, color=INK, linestyle="--", linewidth=1); ax.axvline(hi, color=INK, linestyle="--", linewidth=1)
    ax.set_xlabel("J̃ ricalcolato 300 volte cambiando a caso le liste usate"); ax.set_ylabel("quante volte")
    ax.set_title(f"2. l'incertezza di J̃ per una frase: semi-ampiezza {0.5 * (hi - lo):.2f}", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "figL5_slope.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def figL2_map_illustration():
    """Lay illustration of the mean-field map for one ordered and one disordered formulation of the 1.7B at 4 peers."""
    p = R / "A_qwen1.7b_list_k0_e4_Tan_f0_all_N8" / "gaps.json"
    tabs = {(t["wording"], t["format"]): t for t in json.load(open(p, encoding="utf-8"))["tables"]}
    ms = np.linspace(-1, 1, 201)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(ms, ms, "--", color=INK2, linewidth=1, label="m' = m (no change)")
    picked = []
    for key, t in tabs.items():
        _, ps = policy_from_entries(t["entries"], 1.0)
        F, dF = mf_map(ps, 4)
        slope = float(dF(0.0))
        picked.append((slope, key, np.array([F(m) for m in ms])))
    picked.sort(key=lambda q: q[0])
    lo, hi = picked[0], picked[-1]
    ax.plot(ms, lo[2], "-", color="#2a78d6", label=f"weak formulation, slope at 0 = {lo[0]:.2f}")
    ax.plot(ms, hi[2], "-", color="#eb6834", label=f"strong formulation, slope at 0 = {hi[0]:.2f}")
    ax.set_xlabel("majority now, m (table symmetrised in probability)"); ax.set_ylabel("majority after one round, m'")
    ax.set_title("the map from one round to the next (1.7B, 4 peers)", color=INK, loc="left"); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "figL2_map.png", dpi=180); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    fig1_coupling_vs_exposure(rng); fig2_jump_vs_exposure(); fig3_dF0_classes(); fig4_memory(); fig5_controls()
    fig6_format(rng); figL1_pipeline(); figL2_map_illustration(); figL3_configurations(); figL4_topologies(); figL5_slope()
    print("figures written to", OUT, sorted(p.name for p in OUT.glob("*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
