"""Phase 8 feature maps (protocol v2 section 5.2): every feature set is a function of the CANONICAL 4-view normalised observation
matrix O4 (n, 4: x1 z1 x2 z2) and its mask M4 (n, 4). For sham composites O4 already contains the donor's other-subsystem views.
Feature sets: Bfull, BfullSham{j} (frozen 16-d latents); Fact8 = [A1_8(W1), A2_8(W2)]; Fact8Sham{j} = own A_j_8 + donor-trained
A_o_8d on the donor views; Fact16 = [A1_16, A2_16]; raw_joint = raw masked 4-view window (P7 reference). Encoders are past-only."""
import numpy as np, torch
from experiments.phase5.core import ArmModel, encode_series, raw_features, SUB
from experiments.phase8.core8 import RUNS, init_seed

FEATS = ["Bfull", "BfullSham1", "BfullSham2", "Fact8", "Fact8Sham1", "Fact8Sham2", "Fact16", "raw_joint"]
COMPOSITION = {"Bfull": [("Bfull", [0, 1, 2, 3])], "BfullSham1": [("BfullSham1", [0, 1, 2, 3])], "BfullSham2": [("BfullSham2", [0, 1, 2, 3])],
               "Fact8": [("A1_8", SUB[1]), ("A2_8", SUB[2])], "Fact8Sham1": [("A1_8", SUB[1]), ("A2_8d", SUB[2])],
               "Fact8Sham2": [("A1_8d", SUB[1]), ("A2_8", SUB[2])], "Fact16": [("A1_16", SUB[1]), ("A2_16", SUB[2])]}
SHAM_J = {"BfullSham1": 1, "BfullSham2": 2, "Fact8Sham1": 1, "Fact8Sham2": 2}      # readout target S^sharp; composite input


def load_arm(k, p, r, arm, dev):
    ck = torch.load(f"{RUNS}/ckpt/k{k}_p{p}_r{r}_{arm}_s{init_seed(arm, r)}.pt", map_location=dev, weights_only=False)
    m = ArmModel(len(ck["ins"]), ck["outs"], np.ones((len(ck["ins"]), 8), np.float32), latent=ck["latent"]).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m, ck


class FeatureMap:
    """callable (O4, M4) -> features (n, d); also lag-0 outputs for imputation of the views each encoder decodes."""
    def __init__(self, k, p, r, name, dev):
        self.name, self.dev = name, dev
        self.parts = [] if name == "raw_joint" else [(load_arm(k, p, r, a, dev)[0], cols) for a, cols in COMPOSITION[name]]

    def __call__(self, O4, M4):
        if self.name == "raw_joint":
            return raw_features(O4, M4, [0, 1, 2, 3])
        out = []
        for m, cols in self.parts:
            z, _ = encode_series(m, torch.tensor(O4[:, cols], device=self.dev), torch.tensor(M4[:, cols], device=self.dev))
            out.append(z.cpu().numpy())
        return np.concatenate(out, 1)

    def lag0(self, O4, M4):
        """dict canonical view index -> lag-0 reconstruction (past-only) from the encoder that owns that view."""
        res = {}
        for m, cols in self.parts:
            _, y0 = encode_series(m, torch.tensor(O4[:, cols], device=self.dev), torch.tensor(M4[:, cols], device=self.dev))
            y0 = y0.cpu().numpy()
            for q, c in enumerate(cols):
                res[c] = y0[:, q]
        return res
