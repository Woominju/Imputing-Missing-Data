"""Pre-registered S1R (results/notes/phase6_g2_g3_protocol.md section 10): independent replication of S1 and its kappa = 0
negative control, p = 0.7, arms B1, B2, Sham1, Sham2. Data: joint integration exactly as G2 (gen6 seed rule), ids 30000+."""
import experiments.phase6.core6 as C

R, NT = 24, 8
ROOT = "results/extra/phase6/s1r"


def ids(r):
    return dict(train=30000 + r, train_donor=30100 + r, val=30200 + r, val_donor=30300 + r,
                test=[31000 + 100 * r + i for i in range(NT)], test_donor=[34000 + 100 * r + i for i in range(NT)])


def all_ids():
    out = []
    for r in range(R):
        I = ids(r); out += [(I["train"], 20000), (I["train_donor"], 20000), (I["val"], 5000), (I["val_donor"], 5000)]
        out += [(q, 5000) for q in I["test"] + I["test_donor"]]
    return out


def patch():
    C.norm_stats(0.0); C.norm_stats(1.0)                      # G2 pool constants, computed from the G2 data dir
    C.DDIR = f"{ROOT}/data"; C.ids = ids
    return C
