# v1 — SUPERSEDED (pre-D6 GP optimiser)

Produced 2026-10-27 02:31–02:4x KST with `baselines/fprm/gp.py` **before** fix D6 (restart when L-BFGS-B from the
MATLAB initial point terminates abnormally). In `cond_a/primary_refx_p0.3_s0_*` the x->y fit with E=3,tau=3 stopped
on a large-length-scale plateau (NLL -2444 vs -3570 after the fix). Kept for the record only; use `v2/`.
