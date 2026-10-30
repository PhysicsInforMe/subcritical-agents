# How much do LLM agents actually couple?

Paper: arXiv (submission in preparation; the link will be added here).

A CPU-only study of how strongly small open-weight language-model agents
couple to one another, measured from their logits. For two Qwen3 models
(0.6B and 1.7B parameters) in an abstract two-label game, the policy of an
agent, P(+ | peers, memory), is read once from the logits for every
configuration of the peers it sees, and the population dynamics then runs on
the tabulated policy at zero model cost. A spin model is fitted to the table:
a coupling per peer, a label bias, and a mean-field criterion (the slope of
the map from one round to the next at the disordered point) that says whether
the disordered state is unstable. Three design knobs are varied on a paired
design of ten prompt formulations: the number of peers shown at each update
(2 to 30), the memory of the agent's own past states (0 to 3), and the
format of the peer list.

What the paper reports, in short: the coupling per peer falls as more peers
are shown, in both models; memory of one's own states reduces the coupling to
the peers and adds a persistence term; the list separator does not move the
coupling, the presentation of the peers (list against count) does in the
larger model; the linear criterion used in earlier work overestimates order,
and the paper decomposes the gap exactly at four peers (saturation, shape of
the response, label bias, dispersion between orderings). Every finding has a
rule written before the data and a replication on five wordings written
after the first analysis; every rule and its outcome is in the appendix.

## Layout

- `src/` measures (order parameters, regressions, O-information, damage
  spreading), the logit reader and the tabulation of policies, the tabulated
  dynamics, the graph generators.
- `configs/` one YAML per cell (model, channel, memory, exposure, decoding
  scale, topology, precision, HuggingFace revision) and the shared
  formulation files.
- `scripts/` the tabulation (`phase_a_tabulate.py`, loads the models), the
  dynamics, and every downstream analysis that reads the committed tables;
  `paper_numbers.py` builds the registry of every number the paper quotes,
  `paper_tables.py` and `paper_figures.py` the tables and figures.
- `results/` the raw gap tables (`gaps.json`) of every cell with their
  manifests (revision, precision, seeds, commit hash, hardware), the measure
  files, the analysis outputs, and the launch scripts of the later cells.
- `tests/` unit tests of the physical measures against exact values (2D
  Ising, voter model, synthetic regressions, the mean-field ceiling).
- `paper/arxiv/` the LaTeX source with its generated tables and figures;
  `paper/numbers.json` the registry of every number quoted in the text,
  `paper/tables.md` the generated tables, `paper/paper.pdf` the compiled paper.
- `figures/paper/` the figures of the paper.

## Reproducing

Python 3.12, CPU only; `requirements.txt` pins the versions the analyses were
run with.

```
pip install -r requirements.txt
pytest -o addopts=""                 # measures against exact values
python scripts/reproduce.py --check  # rerun every analysis from the committed
                                     # tables into a copy and compare each file
                                     # with the committed one
python scripts/reproduce.py          # same, overwriting the result files
```

The model runs (tabulation of the policies, the generation check) are not
repeated by the driver: their outputs, the `gaps.json` tables, are its inputs.
To re-tabulate a cell, `python scripts/phase_a_tabulate.py configs/<...>.yaml`
loads the model named in the YAML with `transformers` at the pinned revision
and precision; the 0.6B runs in float32, the 1.7B in bfloat16 with the
precision check reported in the paper.

Section and item numbers quoted in docstrings (for example "design notes
§14.1 B4") refer to the project's internal design notes; `NOTES.md` maps the
ones that matter to the sections of the paper.

## Licence and citation

MIT (see `LICENSE`). If you use the code or the tables, please cite the paper;
the BibTeX entry will be added with the arXiv identifier.
