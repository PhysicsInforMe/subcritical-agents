# Notes on the references in the code

Docstrings and comments quote section and item numbers of the project's
internal design notes, written before the measurements. They are kept as
written because the manifests and the earlier result files refer to them.
The ones that matter for reading the code:

| Reference | Content | Where it is in the paper |
|---|---|---|
| §4.1, §4.2 | order parameters and the battery of measures (mean-field criterion, O-information, susceptibility, damage spreading, hysteresis, response entropy) | Setup, Estimators; Dynamics and controls |
| §4.3 | policies tabulated from the logits, Glauber form logit P = 2(h + J k) | Setup, Tabulated policies and the flip symmetry; Estimators |
| §10.x | reading notes on the cited works (De Nobili, El et al., De Marzo, Flint, Brockers, Okawa) | Related work |
| §14.1 B1-B15, C1-C3 | audit points on the design (ordering sampling B2-B3, decoding scale B4, mean-field criterion B4b, tokenisation B5, finite size B10, covariates B15, precision C3) | Setup; Limitations |
| §14.2 | reproducibility rules (cell identifiers, manifests, seeds) | Reproducibility |
| §16-§25 | results notes by phase and the register of findings | Results; Appendix B |
| revision R1-..., R2-..., R3-..., R4-... | items raised in the revision rounds of the manuscript and the change each one produced | Appendix B lists every declared rule with its outcome |
