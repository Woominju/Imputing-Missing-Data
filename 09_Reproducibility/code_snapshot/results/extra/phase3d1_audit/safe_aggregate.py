"""Phase 3D-1 safe aggregation of the Phase 3C probe results (replaces analyze.py for any conclusion; analyze.py is kept).
Reads the ORIGINAL eval_probe.csv, removes the invalid *_zremoved_refit rows after proving they are the x-removed rows,
checks the result equals recovery/eval_probe_valid_rows.csv, and takes z-removed numbers ONLY from eval_probe_zremoved.csv.
Fails loudly (assert) if any invalid row would reach an output."""
import numpy as np, pandas as pd

P3C = "results/extra/phase3c_hidden_state"; OUT = "results/extra/phase3d1_audit"
INVALID = "_zremoved_refit"


def load_valid_probe():
    raw = pd.read_csv(f"{P3C}/eval_probe.csv")
    bad = raw.label.str.endswith(INVALID)
    zr = raw[bad].sort_values(["model", "seed", "probe"]).test_rmse.to_numpy()
    xr = raw[raw.label.str.endswith("_xremoved_refit")].sort_values(["model", "seed", "probe"]).test_rmse.to_numpy()
    assert bad.sum() == 18 and np.array_equal(zr, xr), "invalid-row signature changed; re-audit before aggregating"
    valid = raw[~bad].reset_index(drop=True)
    rec = pd.read_csv(f"{P3C}/recovery/eval_probe_valid_rows.csv")
    assert valid.shape == rec.shape and (valid.fillna(-999).astype(str).values == rec.fillna(-999).astype(str).values).all()
    zfix = pd.read_csv(f"{P3C}/eval_probe_zremoved.csv")
    assert len(zfix) == 36 and set(zfix["mode"]) == {"refit_on_z_removed", "natural_probe_applied_to_z_removed"}
    return valid, zfix


if __name__ == "__main__":
    valid, zfix = load_valid_probe()
    lat = valid[valid.label.isin(["latent_C-xz", "latent_FDR-xz"])]
    t1 = lat.pivot_table(index=["probe", "seed"], columns="model", values="test_rmse")
    t1["FDR_minus_C"] = t1["FDR-xz"] - t1["C-xz"]
    ctrl = valid[valid.model.isin(["control", "random"]) | valid.vars.eq("x")].pivot_table(index=["label", "probe"], columns="seed", values="test_rmse")
    zt = zfix.pivot_table(index=["model", "mode", "probe"], columns="seed", values="test_rmse")
    out = pd.concat({"latent_xz": t1.reset_index().melt(id_vars=["probe", "seed"]),
                     "controls_and_x_only": ctrl.reset_index().melt(id_vars=["label", "probe"]),
                     "z_removed_CORRECTED": zt.reset_index().melt(id_vars=["model", "mode", "probe"])}, names=["table"]).reset_index(level=0)
    assert not out.astype(str).apply(lambda c: c.str.contains(INVALID)).any().any()
    out.to_csv(f"{OUT}/probe_summary_safe.csv", index=False)
    pd.set_option("display.width", 250)
    print(t1.round(4).to_string()); print(ctrl.round(4).to_string()); print(zt.round(4).to_string())
