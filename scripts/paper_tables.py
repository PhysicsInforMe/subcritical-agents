#!/usr/bin/env python
"""Markdown tables of the paper, generated from paper/numbers.json so that every
cell carries the same rounding as the prose (rule: numbers never transcribed).
Output: paper/tables.md, one section per table, to be pasted or referenced
from the paper sources.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

V = json.load(open("paper/numbers.json", encoding="utf-8"))["values"]


def t(name: str) -> str:
    return V[name]["text"] if name in V else "n/a"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    parts = []
    # Table 1: validation
    rows = [["Ising 2D, critical point from the peaks of χ", t("phase0.ising.Tc"), t("const.ising.Tc_exact")],
            ["Ising 2D, critical point from the Binder crossing", t("phase0.ising.binder_crossing"), t("const.ising.Tc_exact")],
            ["Ising 2D, γ/ν", t("phase0.ising.gamma_over_nu"), t("const.ising.gamma_over_nu_exact")],
            ["Ising 2D, β/ν", t("phase0.ising.beta_over_nu"), t("const.ising.beta_over_nu_exact")],
            ["Voter model, exponent of the consensus time in N (single updates; 1 in rounds, t = N ln 2 on the complete graph)", t("phase0.drift.t_cons_exponent"), "2"],
            ["De Nobili literal prompt, 0.6B, ratio h̃/J̃", t("qwen0.6b.denobili_literal.ratio"), "about 8 (llama, De Nobili)"],
            ["De Nobili literal prompt, 1.7B, ratio h̃/J̃", t("qwen1.7b.denobili_literal.ratio"), "about 8 (llama, De Nobili)"],
            ["Ten A/B formulations, 0.6B, median ratio h̃/J̃", t("qwen0.6b.field_ratio.e4.median"), "above 3 in " + t("qwen0.6b.field_ratio.e4.n_above_3")],
            ["Ten A/B formulations, 1.7B, median ratio h̃/J̃", t("qwen1.7b.field_ratio.e4.median"), "above 3 in " + t("qwen1.7b.field_ratio.e4.n_above_3")]]
    parts.append("## Table 1. Validation of the method\n\n" + table(["Quantity", "Measured [95% interval]", "Reference"], rows))
    # Table 2: grid J_gap, h_gap, zJ by model, channel, k, e
    rows = []
    for m in ("qwen0.6b", "qwen1.7b"):
        for ch in ("list", "count"):
            for k in (0, 3):
                for e in (2, 4, 7):
                    tag = f"{m}.{ch}.k{k}.e{e}"
                    if f"{tag}.J_gap.mean" in V:
                        rows.append([m.replace("qwen", "Qwen3-").upper().replace("QWEN3-", "Qwen3-"), ch, str(k), str(e), t(f"{tag}.J_gap.mean"),
                                     t(f"{tag}.J_gap.sd"), t(f"{tag}.J_gap.ci_halfwidth_within.mean"), t(f"{tag}.h_gap.mean"), t(f"{tag}.zJ_gap.mean"),
                                     t(f"{tag}.J_mem_gap.mean") if k else ""])
    parts.append("## Table 2. The grid at the poor channel: means over the ten formulations\n\n" +
                 table(["Model", "Channel", "Memory k", "Peers e", "J̃ mean", "J̃ sd across formulations", "half-width of the 95% ordering-bootstrap interval of J̃ inside a formulation (no sampling error up to 4 peers)", "h̃ mean", "z̄J̃ mean", "J_mem mean"], rows))
    # Table 3: exposure summary z̄J̃, F'(0) mean and classes
    rows = []
    for m, lab in (("qwen0.6b", "Qwen3-0.6B"), ("qwen1.7b", "Qwen3-1.7B")):
        for e in (2, 4, 7, 10, 30):
            key32 = f"{m}.classes.e{e}_ord32"; key8 = f"{m}.classes.e{e}_ord8"
            key = key32 if f"{key32}.triple" in V else key8
            n_ord = "32" if key == key32 else "8"
            zj = t(f"{m}.ord32.e{e}.zJ_gap.mean32") if f"{m}.ord32.e{e}.zJ_gap.mean32" in V else t(f"{m}.zJ_gap.e{e}")
            rk = key.split(".")[-1]
            rows.append([lab, str(e), n_ord, zj, t(f"{key}.dF0_mean"), t(f"const.mf.ceiling.e{e}"),
                         t(f"{m}.ceiling_ratio.{rk}.mean") + " (at the ceiling: " + t(f"{m}.ceiling_ratio.{rk}.n_marginal") + ")",
                         (t("qwen1.7b.bf16_classes.e4.defensible_triple") + " (bfloat16 error propagated; one formulation marginal by construction)") if (m == "qwen1.7b" and e == 4 and "qwen1.7b.bf16_classes.e4.defensible_triple" in V) else (t(f"{key}.triple") + (" (one of the nine within 0.02 of 1)" if (m == "qwen0.6b" and e == 4) else "")), (t(f"{m}.sectorA32.e{e}.triple") + " (32 orderings, firm in " + t(f"{m}.sectorA32.e{e}.n_firm") + ")") if f"{m}.sectorA32.e{e}.triple" in V else (t(f"{m}.sectorA.e{e}.triple") + (" (8 orderings, exact; one of the bistable formulations is the first-peer-copy policy, marginal by construction: 3 + 1)" if (m == "qwen1.7b" and e == 4) else " (8 orderings, exact)"))])
    parts.append("## Table 3. Total push and mean-field criterion by exposure\n\n" +
                 table(["Model", "Peers e", "Orderings", "z̄J̃ mean", "F'(0) mean", "ceiling E₀\\|k\\|",
                        "F'(0)/ceiling mean (formulations within 0.01 of the ceiling)",
                        "classes ordered/disordered/undetermined (symmetrised map)",
                        "classes bistable/field-dominated/disordered (sector-A map; from 7 peers on the 32-ordering tables, with the number of formulations whose class holds in at least 95% of the ordering-bootstrap draws)"], rows))
    # Table 4: memory curve
    rows = []
    for m, lab in (("qwen0.6b", "Qwen3-0.6B"), ("qwen1.7b", "Qwen3-1.7B")):
        for k in (0, 1, 2, 3):
            rows.append([lab, str(k), t(f"{m}.memory_curve.J_gap.k{k}"), t(f"{m}.memory_curve.J_mem_gap.k{k}") if k else "",
                         t(f"{m}.t4.J.k{k}-k{k-1}") if k else "", t(f"{m}.t4.Jmem_per_state.k{k}-k{k-1}") if k >= 2 else ""])
    parts.append("## Table 4. Memory curve at 4 peers (list channel); intervals are t intervals on the five per-wording means\n\n" +
                 table(["Model", "Memory k", "J̃ to the peers", "J_mem per state (mean)", "paired step J̃(k) − J̃(k−1) [95% t interval]", "paired step of J_mem per state [95% t interval]"], rows))
    # Table 5: format
    rows = []
    for m, lab in (("qwen0.6b", "Qwen3-0.6B"), ("qwen1.7b", "Qwen3-1.7B")):
        for e in (2, 30):
            rows.append([lab, str(e)] + [t(f"{m}.fmt.J_gap.{f}.e{e}") for f in ("comma", "lines", "semicolon", "brackets")])
        rows.append([lab, "LRT p, mixed model with the wording × format component", t(f"{m}.mixed_wf.format.p") + " (format; LR " + t(f"{m}.mixed_wf.format.LR") + " on " + t(f"{m}.mixed_wf.format.df") + " df)",
                     t(f"{m}.mixed_wf.interaction.p") + " (format × exposure; LR " + t(f"{m}.mixed_wf.interaction.LR") + " on " + t(f"{m}.mixed_wf.interaction.df") + " df)", "", ""])
        rows.append([lab, "LRT p, first version without the wording × format component (anticonservative)", t(f"{m}.fmt.lrt.format.p") + " (format)", t(f"{m}.fmt.lrt.interaction.p") + " (format × exposure)", "", ""])
    parts.append("## Table 5. Format of the peer list\n\n" + table(["Model", "Peers e", "comma", "lines", "semicolon", "brackets"], rows))
    # Table 6: literature comparison
    rows = [["De Nobili, llama, 4 neighbours (fit weighted by the inverse decoding scale, values at the low end of that scan)", "J̃ at most " + t("const.lit.denobili.J_max") + ", h̃ about " + t("const.lit.denobili.h_llama"), "total push at most " + t("const.lit.denobili.zJ_llama_max"), ""],
            ["Ours, 0.6B, 4 peers, ten A/B formulations", "J̃ " + t("qwen0.6b.list.k0.e4.J_gap.mean"), "z̄J̃ " + t("qwen0.6b.zJ_gap.e4"), "F'(0) " + t("qwen0.6b.mf8.e4.dF0.mean") + ", above 1 in " + t("qwen0.6b.mf8.e4.dF0.n_above_1") + " (one more within " + t("f0dec.margin") + " of 1; on the odd part of the table alone " + t("qwen0.6b.f0dec.F0_odd.mean") + ", above 1 in " + t("qwen0.6b.f0dec.F0_odd.n_above_1") + " with " + t("qwen0.6b.f0dec.F0_odd.n_within_margin") + " at the threshold, Appendix C)"],
            ["Ours, 1.7B, 4 peers, ten A/B formulations", "J̃ " + t("qwen1.7b.list.k0.e4.J_gap.mean"), "z̄J̃ " + t("qwen1.7b.zJ_gap.e4"), "F'(0) " + t("qwen1.7b.mf8.e4.dF0.mean") + ", above 1 in " + t("qwen1.7b.mf8.e4.dF0.n_above_1") + " (point estimates, three of them within " + t("f0dec.margin") + " of 1 including the first-peer-copy policy at 1.000; with the bfloat16 error propagated " + t("qwen1.7b.bf16_classes.e4.defensible_triple") + " ordered/disordered/undetermined, Table 3)"],
            ["Ours, 0.6B, De Nobili literal prompt", "J̃ " + t("qwen0.6b.denobili_literal.J") + ", h̃ " + t("qwen0.6b.denobili_literal.h"), "", ""],
            ["Ours, 1.7B, De Nobili literal prompt", "J̃ " + t("qwen1.7b.denobili_literal.J") + ", h̃ " + t("qwen1.7b.denobili_literal.h"), "", ""],
            ["De Marzo, 49 peers, most models (every agent sees all the others: no sampling ceiling on β)", "", "", "β > 1 (β is the slope of the tanh at m = 0, the analogue of F'(0))"],
            ["Ours, 30 peers, 32 orderings (the only row comparable with De Marzo: ceiling " + t("const.mf.ceiling.e30") + ")", "", "z̄J̃ " + t("qwen0.6b.ord32.e30.zJ_gap.mean32") + " (0.6B), " + t("qwen1.7b.ord32.e30.zJ_gap.mean32") + " (1.7B)",
             "F'(0) " + t("qwen0.6b.ord32.e30.dF0_sym.mean32") + " (0.6B), " + t("qwen1.7b.ord32.e30.dF0_sym.mean32") + " (1.7B)"]]
    scale = ["scan (his fit is weighted by the inverse scale; values at the low end)", "1 (analytic rescaling)", "1 (analytic rescaling)", "1 (analytic rescaling)", "1 (analytic rescaling)",
             t("const.lit.demarzo.T") + " (sampling)", "1 (analytic rescaling)"]
    rows = [r + [sc] for r, sc in zip(rows, scale)]
    parts.append("## Table 6. Comparison with the published estimates, on two distinct quantities\n\n" + table(["Setting", "Coupling and field", "Total push", "Mean-field criterion", "Decoding scale at which the policy is read"], rows))
    # Table 7: precision and orderings checks
    rows = [["bfloat16 vs float32, 1.7B, 2 peers, ten formulations (" + t("fp32check.qwen1.7b.e2.n_gaps") + " gaps)", t("fp32check.qwen1.7b.e2.diff.mean"), t("fp32check.qwen1.7b.e2.diff.sd"), t("fp32check.qwen1.7b.e2.diff.max_abs"), t("fp32check.qwen1.7b.e2.corr")],
            ["bfloat16 vs float32, 1.7B, 30 peers, one formulation (wording 0, comma; " + t("fp32check.qwen1.7b.e30.n_gaps") + " gaps)", t("fp32check.qwen1.7b.e30.diff.mean"), t("fp32check.qwen1.7b.e30.diff.sd"), t("fp32check.qwen1.7b.e30.diff.max_abs"), t("fp32check.qwen1.7b.e30.corr")],
            ["bfloat16 vs float32, 0.6B, 2 peers, ten formulations (" + t("bf16.n_gaps") + " gaps)", t("bf16.diff.mean"), t("bf16.diff.sd"), t("bf16.diff.max_abs"), t("bf16.corr")]]
    p1 = "## Table 7. Numerical precision of the tabulation (difference float32 minus bfloat16, logit)\n\n" + table(["Cells", "mean", "sd", "max |diff|", "correlation"], rows)
    rows = []
    for m, lab in (("qwen0.6b", "Qwen3-0.6B"), ("qwen1.7b", "Qwen3-1.7B")):
        for e in (7, 10, 30):
            tag = f"{m}.ord32.e{e}"
            if f"{tag}.zJ_gap.mean8" in V:
                rows.append([lab, str(e), t(f"{tag}.zJ_gap.mean8") + " → " + t(f"{tag}.zJ_gap.mean32"), t(f"{tag}.zJ_gap.diff32_8"), t(f"{tag}.zJ_gap.inside"),
                             t(f"{tag}.dF0_sym.mean8") + " → " + t(f"{tag}.dF0_sym.mean32"), t(f"{tag}.dF0_sym.diff32_8"), t(f"{tag}.dF0.class_flips")])
    p2 = "## Table 8. Eight against thirty-two orderings per configuration\n\n" + table(["Model", "Peers e", "z̄J̃ 8 → 32", "paired diff [95%]", "32 inside CI(8)", "F'(0) 8 → 32", "paired diff [95%]", "class flips (F'(0) > 1)"], rows)
    parts += [p1, p2]
    # Table 9: the ten formulations at 4 peers, no memory
    rows = []
    for m, lab in (("qwen0.6b", "Qwen3-0.6B"), ("qwen1.7b", "Qwen3-1.7B")):
        for w in range(5):
            for f in ("comma", "lines"):
                tag = f"{m}.formulation.w{w}_{f}"
                if f"{tag}.J_gap" in V:
                    rows.append([lab, str(w), f, t(f"{tag}.J_gap"), t(f"{tag}.h_gap"), t(f"{tag}.nll")])
        for w in range(5):
            for f in ("comma", "lines"):
                tag = f"{m}.nw.formulation.w{w}_{f}"
                if f"{tag}.J_gap" in V:
                    rows.append([lab, str(w + 5) + " (new)", f, t(f"{tag}.J_gap"), t(f"{tag}.h_gap"), ""])
    parts.append("## Table 9. The twenty formulations at 4 peers without memory (five original wordings and five new ones, two separators): coupling, label bias and mean prompt loss (the loss is computed for the five original wordings only)\n\n" +
                 table(["Model", "Wording", "Format", "J̃", "h̃ (positive: toward A)", "mean token loss of the prompt (NLL)"], rows))
    # Table 10: decomposition of e J~ - F'(0) at 4 peers (round 5, 2026-10-24)
    if "qwen0.6b.f0dec.eJ.mean" in V:
        labels = [("eJ", "linear criterion e·J̃"), ("F0_add", "additive, no bias: σ(2J̃k)"),
                  ("F0_add_bias", "additive with bias, symmetrised in probability"),
                  ("F0_odd", "odd part of the table in logit space (bias removed)"),
                  ("F0_even", "mean gap with its even part, symmetrised in probability"),
                  ("F0_disp", "dispersion between orderings, no bias"),
                  ("F0_paper", "the paper's F'(0): mean of probabilities over orderings, symmetrised in probability")]
        rows = []
        for key, lab in labels:
            rows.append([lab] + [t(f"{m}.f0dec.{key}.mean") + " (" + t(f"{m}.f0dec.{key}.triple") + ")" for m in ("qwen0.6b", "qwen1.7b")])
        parts.append("## Table 10. Decomposition of the gap between the linear criterion and the mean-field slope at 4 peers, k = 0 (means over the ten formulations; in brackets the formulations above 1, within " + t("f0dec.margin") + " of 1, and below 1)\n\n" +
                     table(["Policy the slope is computed on", "Qwen3-0.6B", "Qwen3-1.7B"], rows))
        rows = []
        for m, lab in (("qwen0.6b", "Qwen3-0.6B"), ("qwen1.7b", "Qwen3-1.7B")):
            g = f"{m}.f0dec"
            rows.append([lab, t(f"{g}.step.saturation"), t(f"{g}.step.shape_k"),
                         t(f"{g}.step.bias_then_disp.bias") + " / " + t(f"{g}.step.disp_then_bias.bias"),
                         t(f"{g}.step.bias_then_disp.dispersion") + " / " + t(f"{g}.step.disp_then_bias.dispersion"),
                         t(f"{g}.step.interaction"), t(f"{g}.residual")])
        parts.append("## Table 10b. Sequential steps of the decomposition (means over the ten formulations; bias and dispersion in the order bias-then-dispersion / dispersion-then-bias)\n\n" +
                     table(["Model", "saturation", "shape in k", "bias (even part)", "dispersion between orderings", "interaction (bias step with the dispersion present minus bias step alone)", "residual after saturation"], rows))
        rows = []
        for m, lab in (("qwen0.6b", "Qwen3-0.6B"), ("qwen1.7b", "Qwen3-1.7B")):
            for w in range(5):
                for f in ("comma", "lines"):
                    rt = f"{m}.f0dec.w{w}_{f}"
                    rows.append([lab, str(w), f, t(f"{rt}.h"), t(f"{rt}.eJ"), t(f"{rt}.F0_add"), t(f"{rt}.F0_odd"), t(f"{rt}.F0_even"), t(f"{rt}.F0_paper"), t(f"{rt}.noise.std_F0")])
        parts.append("## Table 10c. The same decomposition per formulation at 4 peers (three decimals; the last column is the standard deviation of F'(0) under a Gaussian perturbation of every gap with the precision difference of Table 7)\n\n" +
                     table(["Model", "Wording", "Format", "h̃", "e·J̃", "additive", "odd part", "with even part", "paper", "std F'(0)"], rows))
    # LaTeX tables for paper/arxiv/main.tex (revision R4-11): the same numbers, no transcription
    def tex_esc(x: str) -> str:
        return x.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")
    tex_rows = []
    for m, lab in (("qwen0.6b", "Qwen3-0.6B"), ("qwen1.7b", "Qwen3-1.7B")):
        for e in (2, 4, 7, 10, 30):
            key32 = f"{m}.classes.e{e}_ord32"; key = key32 if f"{key32}.triple" in V else f"{m}.classes.e{e}_ord8"
            rk = key.split(".")[-1]
            zj = t(f"{m}.ord32.e{e}.zJ_gap.mean32") if f"{m}.ord32.e{e}.zJ_gap.mean32" in V else t(f"{m}.zJ_gap.e{e}")
            sym = (t("qwen1.7b.bf16_classes.e4.defensible_triple") + "$^{*}$") if (m == "qwen1.7b" and e == 4 and "qwen1.7b.bf16_classes.e4.defensible_triple" in V) else (t(f"{key}.triple") + ("$^{\\ddagger}$" if (m == "qwen0.6b" and e == 4) else ""))
            sec = t(f"{m}.sectorA32.e{e}.triple") if f"{m}.sectorA32.e{e}.triple" in V else (t(f"{m}.sectorA.e{e}.triple") + ("$^{\\dagger}$" if (m == "qwen1.7b" and e == 4) else ""))
            tex_rows.append(f"{lab} & {e} & {zj} & {t(key + '.dF0_mean')} & {t('const.mf.ceiling.e%d' % e)} & {t(m + '.ceiling_ratio.' + rk + '.mean')} ({t(m + '.ceiling_ratio.' + rk + '.n_marginal')}) & {sym} & {sec} \\\\")
    Path("paper/arxiv/tab_exposure.tex").write_text("% generated by scripts/paper_tables.py from paper/numbers.json; do not edit\n\\begin{tabular}{llcccccc}\n\\toprule\nModel & Peers & $\\bar z\\tilde J$ & $F'(0)$ & ceiling & ratio (at ceiling) & sym. classes & sector-A classes \\\\\n\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    tex_rows = []
    for m, lab in (("qwen0.6b", "0.6B"), ("qwen1.7b", "1.7B")):
        for w in range(5):
            for f in ("comma", "lines"):
                tag = f"{m}.formulation.w{w}_{f}"
                tex_rows.append(f"{lab} & {w} & {f} & {t(tag + '.J_gap')} & {t(tag + '.h_gap')} & {t(tag + '.nll')} \\\\")
        for w in range(5):
            for f in ("comma", "lines"):
                tag = f"{m}.nw.formulation.w{w}_{f}"
                if f"{tag}.J_gap" in V:
                    tex_rows.append(f"{lab} & {w + 5} (new) & {f} & {t(tag + '.J_gap')} & {t(tag + '.h_gap')} & \\\\")
    Path("paper/arxiv/tab_formulations.tex").write_text("% generated by scripts/paper_tables.py from paper/numbers.json; do not edit\n\\begin{tabular}{llcccc}\n\\toprule\nModel & Wording & Format & $\\tilde J$ & $\\tilde h$ & token loss \\\\\n\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    if "qwen0.6b.f0dec.eJ.mean" in V:
        tex_rows = []
        for key, lab in (("eJ", "linear criterion $e\\tilde J$"), ("F0_add", "additive, no bias, $\\sigma(2\\tilde J k)$"),
                         ("F0_add_bias", "additive with bias, symmetrised in probability"),
                         ("F0_odd", "odd part of the table in logit space"),
                         ("F0_even", "mean gap with its even part, symmetrised in probability"),
                         ("F0_disp", "dispersion between orderings, no bias"),
                         ("F0_paper", "$F'(0)$ of the paper (mean of probabilities over orderings)")):
            tex_rows.append(f"{lab} & {t('qwen0.6b.f0dec.' + key + '.mean')} ({t('qwen0.6b.f0dec.' + key + '.triple')}) & {t('qwen1.7b.f0dec.' + key + '.mean')} ({t('qwen1.7b.f0dec.' + key + '.triple')}) \\\\")
        tex_rows.append("\\midrule")
        for m, lab in (("qwen0.6b", "0.6B"), ("qwen1.7b", "1.7B")):
            g = f"{m}.f0dec"
            tex_rows.append(f"\\multicolumn{{3}}{{p{{0.96\\linewidth}}}}{{steps, {lab}: saturation {t(g + '.step.saturation')}, shape in $k$ {t(g + '.step.shape_k')}, bias {t(g + '.step.bias_then_disp.bias')} / {t(g + '.step.disp_then_bias.bias')}, dispersion {t(g + '.step.bias_then_disp.dispersion')} / {t(g + '.step.disp_then_bias.dispersion')} (bias-then-dispersion / dispersion-then-bias), interaction {t(g + '.step.interaction')} (bias step with the dispersion present minus bias step alone)}} \\\\")
        tex_rows = [re.sub(r"(?<![\w$])-(?=\d)", "$-$", row) for row in tex_rows]
    Path("paper/arxiv/tab_decomposition.tex").write_text("% generated by scripts/paper_tables.py from paper/numbers.json; do not edit\n\\begin{tabular}{p{0.56\\linewidth}cc}\n\\toprule\nPolicy the slope is computed on & Qwen3-0.6B & Qwen3-1.7B \\\\\n\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    Path("paper/tables.md").write_text("# Tables of the paper (generated by scripts/paper_tables.py from paper/numbers.json)\n\n" + "\n\n".join(parts) + "\n", encoding="utf-8")
    print("paper/tables.md written,", len(parts), "tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
