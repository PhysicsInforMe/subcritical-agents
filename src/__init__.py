"""Subcritical regime of agentic architectures: measurement library.

Layout (design notes §6):
    src.ising      2D Ising model for the Phase 0 calibration
    src.measures   order parameters, susceptibilities, Binder, regressions,
                   information measures, damage spreading, hysteresis
    src.dynamics   Glauber dynamics on a tabulated policy, voter model
    src.topology   interaction graphs as adjacency structures
    src.llm        logit extraction and policy tabulation (Phase 0 check D)
    src.common     manifests and reproducibility helpers
"""
