"""Phase 6 G1 analysis (CPU), as fixed in results/notes/phase6_g1_protocol.md section 4.
Unit = independent test trajectory (32). Ratio estimators sum/n per stratum (pooled over the 10 test masks); cluster bootstrap
over trajectories (10,000 resamples): one-sided 95% lower bound for certificates G_L0(f) = E[y^2 - (y - f)^2 | L0], two-sided
95% percentile CI otherwise. y is in units of sd(y) (not centred; E[y] = 0 exactly by symmetry).
Writes results/extra/phase6/g1/analysis_g1.csv and prints the main table."""
import glob, numpy as np, pandas as pd

O = "results/extra/phase6/g1"; NB = 10000


def ratio_boot(S, N, rng, idx):
    """S, N: (n_traj,) sums and counts; returns point estimate and bootstrap draws of sum(S)/sum(N)."""
    est = S.sum() / max(N.sum(), 1)
    b = S[idx].sum(1) / np.maximum(N[idx].sum(1), 1)
    return est, b


def main():
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{O}/k*_p*.csv")) if not f.endswith("analysis_g1.csv")], ignore_index=True)
    out = []
    for (k, p), X in D.groupby(["kappa", "p"]):
        trajs = np.sort(X.traj.unique()); rng = np.random.default_rng(0); idx = rng.integers(0, len(trajs), (NB, len(trajs)))
        for st, Y in X.groupby("stratum"):
            W = Y.groupby(["quantity", "traj"])[["sum", "n"]].sum()
            get = lambda q: (W.loc[q]["sum"].reindex(trajs).fillna(0).values, W.loc[q]["n"].reindex(trajs).fillna(0).values)
            qs = set(Y.quantity)
            def add(name, est, b, kind="ci"):
                lo, hi = (np.percentile(b, 5), np.inf) if kind == "lcb" else (np.percentile(b, 2.5), np.percentile(b, 97.5))
                out.append(dict(kappa=k, p=p, stratum=st, quantity=name, estimate=est, lo=lo, hi=hi, bound=kind, n=int(get("y2")[1].sum())))
            Sy, N = get("y2"); ey2, by2 = ratio_boot(Sy, N, rng, idx); add("E[y2]", ey2, by2)
            for kind in ("U", "UV", "UVc", "UVs"):
                for suf in ("avg", "A"):
                    s, _ = get(f"risk_{kind}_{suf}"); e, b = ratio_boot(s, N, rng, idx); add(f"R({kind}_{suf})", e, b)
                s, _ = get(f"cert_{kind}"); e, b = ratio_boot(s, N, rng, idx); add(f"G({kind})", e, b, "lcb" if st.startswith("L0") else "ci")
            sU, _ = get("risk_U_avg")
            for kind in ("UV", "UVc", "UVs"):
                s, _ = get(f"risk_{kind}_avg"); e, b = ratio_boot(sU - s, N, rng, idx); add(f"R(U)-R({kind})", e, b)
                for pre in ("", "raw"):
                    sT, _ = get(f"T{pre}_{kind}"); sD, _ = get(f"Dq{pre}_{kind}")
                    eT, bT = ratio_boot(sT, N, rng, idx); eD, bD = ratio_boot(sD, N, rng, idx)
                    add(f"T{pre}({kind})", eT, bT); add(f"W{pre}({kind})", max(eT, 0) ** 2 / eD, np.maximum(bT, 0) ** 2 / bD)
            for reg in ("p5like", "fixedmask"):
                if f"risk_U_{reg}" in qs:
                    a, _ = get(f"risk_U_{reg}"); c, _ = get(f"risk_UV_{reg}")
                    for nm, s in ((f"R(U_{reg})", a), (f"R(UV_{reg})", c), (f"R(UV_{reg})-R(U_{reg})", c - a)):
                        e, b = ratio_boot(s, N, rng, idx); add(nm, e, b)
            if "risk_UV_A" in qs:
                a, _ = get("risk_U_A"); c, _ = get("risk_UV_A"); e, b = ratio_boot(c - a, N, rng, idx); add("R(UV_A)-R(U_A)", e, b)
            sb, Nb = get("big")
            for nm in ("signerr_U", "signerr_UV"):
                s, _ = get(nm); e, b = ratio_boot(s, sb, rng, idx); add(nm, e, b)
    R = pd.DataFrame(out); R.to_csv(f"{O}/analysis_g1.csv", index=False)
    # certificate summary: Delta* >= p^8 * LCB(G_L0(UV))
    C = R[(R.stratum == "L0") & R.quantity.isin(["G(UV)", "G(UVc)", "G(UVs)"])].copy(); C["p8"] = C.p ** 8
    C["overall_lower"] = C.p8 * np.maximum(C.lo, 0)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    f = lambda v: f"{v:.4f}"
    print("== L0 certificates (one-sided 95% LCB = lo) ==\n", C[["kappa", "p", "quantity", "estimate", "lo", "n", "overall_lower"]].to_string(index=False, float_format=f))
    sel = R.quantity.isin(["E[y2]", "R(U_avg)", "R(UV_avg)", "R(UVc_avg)", "R(UVs_avg)", "R(U)-R(UV)", "R(U)-R(UVc)", "R(U)-R(UVs)", "T(UV)", "T(UVc)", "T(UVs)", "W(UV)", "W(UVs)"])
    print(R[sel & R.stratum.isin(["all", "L0", "L1", "L2+"])].pivot_table(index=["kappa", "p", "stratum"], columns="quantity", values="estimate").to_string(float_format=f))


if __name__ == "__main__":
    main()
