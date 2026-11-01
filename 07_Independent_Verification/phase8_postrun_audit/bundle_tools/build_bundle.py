#!/usr/bin/env python
"""Build the Phase 8 post-run verification bundle (minimal, uploadable) + a full-data SHA-256 list (files NOT in the bundle).
Read-only on the project; writes only under results/audit/. usage: python -I build_bundle.py --root <root>"""
import argparse, glob, hashlib, os, shutil, subprocess, sys, zipfile
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); A = ap.parse_args(); os.chdir(A.root)
AD = "results/audit/phase8_postrun_20261031"; EV = "results/extra/phase8/eval"; AN = "results/extra/phase8/analysis/complete_20261031-181946_0f246de749"
NAME = "phase8_postrun_verification_bundle_20261031"; B = f"results/audit/{NAME}"
if os.path.exists(B):
    shutil.rmtree(B)
os.makedirs(f"{B}/bundle_inputs")


def sha(f):
    h = hashlib.sha256(); h.update(open(f, "rb").read()); return h.hexdigest()


# replicate-level inputs: SSE summed over test trajectories x masks inside each replicate (the confirmatory unit)
acc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_acc.csv.gz"))])
acc.groupby(["kappa", "p", "rep", "pipeline", "stratum", "coord"])[["sse", "n"]].sum().reset_index().to_csv(f"{B}/bundle_inputs/rep_acc.csv.gz", index=False)
imp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_imp.csv.gz"))])
imp.groupby(["kappa", "p", "rep", "method", "view", "stratum"])[["sse", "n"]].sum().reset_index().to_csv(f"{B}/bundle_inputs/rep_imp.csv.gz", index=False)
p7 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/p7imp_*.csv.gz"))])
p7.groupby(["kappa", "p", "rep", "method", "view", "stratum"])[["sse", "n"]].sum().reset_index().to_csv(f"{B}/bundle_inputs/rep_p7imp.csv.gz", index=False)
pair = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_pair.csv.gz"))])
pair.groupby(["kappa", "p", "rep", "pipeline", "pair", "coord"]).agg(sse=("sse", "sum"), n_per_traj=("n", "first"), n_traj=("traj", "nunique")).reset_index().to_csv(f"{B}/bundle_inputs/rep_pair.csv.gz", index=False)
pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_lat.csv.gz"))]).to_csv(f"{B}/bundle_inputs/rep_lat.csv.gz", index=False)
for f in glob.glob(f"{EV}/knn*_k*.csv") + glob.glob(f"{EV}/d5b_k*.csv") + glob.glob("results/extra/phase8/oracle/sat_*.json"):
    shutil.copy(f, f"{B}/bundle_inputs/")
open(f"{B}/bundle_inputs/SOURCE_FILES_sha256.txt", "w").write("".join(f"{sha(f)}  {f}\n" for f in sorted(glob.glob(f"{EV}/*"))))
# documents, scripts, outputs, freeze evidence, analysis
for f in ["results/notes/phase8_postrun_independent_audit_20261031.md", "results/notes/phase8_mathematical_conclusions_20261031.md",
          "results/notes/phase8_mathematical_protocol_v2.md", "results/notes/phase8_revision_crosswalk.md", "results/notes/phase8_final_precheck.md",
          "results/extra/phase8/PROTOCOL_FREEZE_p8.txt", "results/extra/phase8/FREEZE_CHANGES.md", "research_handover_latest.md"]:
    os.makedirs(f"{B}/docs", exist_ok=True); shutil.copy(f, f"{B}/docs/")
for f in ["experiments/phase8/analyze8.py", "experiments/phase8/evaluate8.py", "experiments/phase8/core8.py", "experiments/phase8/feats8.py",
          "experiments/phase8/oracle8.py", "experiments/phase8/eval_p7imp8.py", "experiments/phase8/sched8.py", "experiments/phase8/launch_p8.sh"]:
    os.makedirs(f"{B}/frozen_code_reference", exist_ok=True); shutil.copy(f, f"{B}/frozen_code_reference/")
shutil.copytree(AN, f"{B}/analysis_complete_20261031-181946_0f246de749")
for d in ("out", "logs", "freeze_evidence"):
    shutil.copytree(f"{AD}/{d}", f"{B}/audit_{d}")
os.makedirs(f"{B}/audit_scripts")
for f in glob.glob(f"{AD}/*.py") + glob.glob(f"{AD}/bundle_tools/*.py") + [f"{AD}/README.md"]:
    shutil.copy(f, f"{B}/audit_scripts/")
# self-test: the standalone recompute must reproduce the analysis decisions from the bundle tables alone
r = subprocess.run([sys.executable, "-I", "audit_scripts/recompute_from_bundle.py", "bundle_inputs"], cwd=B, capture_output=True, text=True)
open(f"{B}/audit_logs/recompute_from_bundle_selftest.txt", "w").write(r.stdout + r.stderr); print(r.stdout[-2500:], r.stderr[-1500:])
rc = pd.read_csv(f"{B}/recomputed_confirmatory.csv"); os.replace(f"{B}/recomputed_confirmatory.csv", f"{B}/audit_out/recomputed_confirmatory_from_bundle.csv")
# manifest + zip
files = sorted(os.path.relpath(os.path.join(dp, f), B) for dp, _, fs in os.walk(B) for f in fs)
open(f"{B}/SHA256SUMS.txt", "w").write("".join(f"{sha(os.path.join(B, f))}  {f}\n" for f in files))
Z = f"results/audit/{NAME}.zip"
with zipfile.ZipFile(Z, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files + ["SHA256SUMS.txt"]:
        z.write(os.path.join(B, f), f"{NAME}/{f}")
print("ZIP", Z, os.path.getsize(Z), "bytes", sha(Z))
# full-data list (not bundled): every Phase 8 artefact with size and SHA-256
with open(f"results/audit/{NAME}_FULL_DATA_LIST.txt", "w") as fh:
    fh.write("# Phase 8 artefacts NOT in the minimal bundle (sha256  size  path). Server: /home/minju/faicpaston/Imputing_Missing_Data\n")
    for f in sorted(glob.glob("results/extra/phase8/**/*", recursive=True)):
        if os.path.isfile(f):
            fh.write(f"{sha(f)}  {os.path.getsize(f)}  {f}\n")
print("FULL LIST written")
