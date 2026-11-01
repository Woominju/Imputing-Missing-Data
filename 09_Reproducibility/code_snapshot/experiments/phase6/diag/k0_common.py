"""POST-HOC DIAGNOSTIC (written 2026-10-28 19:30, after the pre-registered G2 analysis showed the kappa = 0, p = 0.7 negative
control B - Sham = +0.0062, 95% CI [0.0022, 0.0101], 20/24 positive). Not part of the pre-registered family.
Question: is the asymmetry caused by the JOINT 6-D adaptive integration (shared step-size control couples the numerical
solutions of the two kappa = 0 subsystems), or by something else (pipeline asymmetry / chance)?
Design: 24 NEW replicates at kappa = 0 (ids 20000+, disjoint from all earlier ids, same seed rule as gen6). Each trajectory is
generated twice from the SAME initial condition: J = joint 6-D integration (src.lorenz.generate_coupled_lorenz, as G2);
S = each 3-D subsystem integrated separately (src.lorenz.integrate with lorenz_rhs, same solver/tolerances/dt/transient).
Arms B1, B2, Sham1, Sham2 trained with the unchanged train6 loop; evaluation = G2 y_j MLP probes (seeds 0, 1), stratum all."""
import numpy as np
import experiments.phase6.core6 as C
from experiments.phase6.gen6 import s0_of

R, NT = 24, 8
ROOT = "results/extra/phase6/diag_k0"


def ids(r):
    return dict(train=20000 + r, train_donor=20100 + r, val=20200 + r, val_donor=20300 + r,
                test=[21000 + 100 * r + i for i in range(NT)], test_donor=[24000 + 100 * r + i for i in range(NT)])


def all_ids():
    out = []
    for r in range(R):
        I = ids(r); out += [(I["train"], 20000), (I["train_donor"], 20000), (I["val"], 5000), (I["val_donor"], 5000)]
        out += [(q, 5000) for q in I["test"] + I["test_donor"]]
    return out


def patch(version):
    """point core6 at the diagnostic data; keep the G2 kappa = 0 normalisation (pool 900..903, joint integration)."""
    C.norm_stats(0.0)
    C.DDIR = f"{ROOT}/data_{version}"; C.ids = ids
    return C
