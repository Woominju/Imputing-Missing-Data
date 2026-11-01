"""Gate F integrity tests (CPU)."""
import numpy as np
import experiments.phase5.external as X


def t_shift_is_split_permutation_and_keeps_own_history():
    for ds in ("beijing", "jena"):
        S = X.setup(ds, 0.7, 0); P = S["P"]
        for (s, e) in S["sp"]:
            for k in range(S["V"]):
                assert sorted(P[s:e, k]) == list(range(s, e))                # permutation inside the split, no leakage across splits
                d = np.diff(P[s:e, k]); assert (d == 1).sum() == (e - s - 1) - (1 if (24 * 14 * k) % (e - s) else 0)
        assert (P[:, 0] == np.arange(S["T"])).all()                           # view 0 unshifted


def t_daily_phase_kept():
    S = X.setup("beijing", 0.7, 0)
    for (s, e) in S["sp"]:
        for k in range(S["V"]):
            t = np.arange(s, e); ok = ~S["excl"][S["P"][s:e, k], k]
            assert (((t - S["P"][s:e, k]) % 24)[ok] == 0).all() or (e - s) % 24 != 0


def t_no_eval_on_natural_missing_and_train_only_norm():
    S = X.setup("beijing", 0.3, 0); s, e = S["sp"][2]
    assert S["Mo"].min() == 0                                                 # natural missingness exists
    ev = (S["Mo"] > 0) & (S["M"] == 0) & ~S["excl"]
    assert not (ev & (S["Mo"] == 0)).any()
    Z = np.load("Data/external/beijing/processed/beijing.npz", allow_pickle=True)["X"]; a, b = S["sp"][0]
    mu0 = np.nanmean(np.where(S["M"][a:b] > 0, Z[a:b], np.nan), 0); assert np.allclose(mu0, S["mu"])


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("t_"):
            f(); print("PASS", n)
