# Phase 8 post-run independent verification (2026-10-31)

Read-only audit of Phase 8 Gate D (complete analysis `results/extra/phase8/analysis/complete_20261031-181946_0f246de749`).
- No training, no GPU, no change to any Phase 7/8 file. Every output of this audit is under `results/audit/`.
- Reports:
  - `results/notes/phase8_postrun_independent_audit_20261031.md`
  - `results/notes/phase8_mathematical_conclusions_20261031.md`

## Scripts (run from the project root; Python of env `reef_gpu`; CPU only: `CUDA_VISIBLE_DEVICES=""`)

| script | independence | what |
|---|---|---|
| `verify_p8_independent.py` | imports no project module; torch only to read O_raw state dicts | L1: re-derives every Family I–IV statistic, gate and sequence from the replicate-level SSE sums and compares with the 18 analysis CSVs. L2: regenerates the 3,840 test masks (own seed_of/MCAR), the strata and normalisation, sym0/zero/LOCF SSE, and LB from the raw pool trajectories. L3: O_raw re-inference (own 64-d window features + numpy MLP forward) for p = 0.7 |
| `audit_execution.py` | no project module | 1,276 expected outputs rebuilt from the protocol; scheduler-log ledger (duplicates, kills, rc); failure/tmp files; trajectory identity and exact-state leakage; GPU/driver per training |
| `verify_p8_supplement.py` | no project module | S1 identical point sets across pipelines; S2 latent-NN paired CIs; S3 Family III by own-L stratum at both κ; S4 G_tot shares; S5 readout-level mask variance |
| `verify_p8_oown.py` | no project module | protocol §5.5 supplementary item 4: oracle-level cross information (O_own_j vs O_raw), which analyze8 does not output |
| `posthoc_driver_sensitivity.py` | no project module | **POST-HOC** sensitivity of the p = 0.7 confirmatory contrasts to the NVIDIA driver change (never replaces a confirmatory result) |
| `bundle_tools/recompute_from_bundle.py` | standalone (numpy/pandas/scipy) | recomputes all confirmatory decisions from the bundle tables alone (for reviewers without server access) |
| `bundle_tools/build_bundle.py` | — | builds the ZIP, `SHA256SUMS.txt` and the full-data list |

```
PY=/home/minju/.conda/envs/reef_gpu/bin/python; AD=results/audit/phase8_postrun_20261031
CUDA_VISIBLE_DEVICES="" $PY -I $AD/verify_p8_independent.py --root $PWD --out $AD/out      # ~10 min on 16 threads
CUDA_VISIBLE_DEVICES="" $PY -I $AD/audit_execution.py      --root $PWD --out $AD/out
CUDA_VISIBLE_DEVICES="" $PY -I $AD/verify_p8_supplement.py --root $PWD --out $AD/out
CUDA_VISIBLE_DEVICES="" $PY -I $AD/verify_p8_oown.py       --root $PWD --out $AD/out
CUDA_VISIBLE_DEVICES="" $PY -I $AD/posthoc_driver_sensitivity.py --root $PWD --out $AD/out
$PY -I $AD/bundle_tools/build_bundle.py --root $PWD
```

## Result
- **54/54 checks OK** (`out/verification_checks.csv`, log `logs/verify_run_stdout.txt`).
- Input SHA-256s: `out/input_sha256.txt`. Freeze evidence (Gate C originals byte-identical to their frozen hashes, current versions, diffs): `freeze_evidence/`.

## Limits (see audit report §C1, §K)
- R1 readouts were never saved, so R1-side SSE is verified only at L1 (from the evaluate8 sums). Re-deriving it would need re-fitting, which counts as training and was not done.
- O_lat-side SSE is likewise verified at L1 only. A re-inference spot check would need the project encoder code and was not done (`reinfer_olat_spotcheck.py`, named in the docstring, was not created).
- The Bernstein auxiliary bound of protocol §4.5 (M-1) was never implemented by the project and is not recomputed here.
- Reproducing GPU numerics under the old and new driver is not possible without re-running, so this is unverifiable.
