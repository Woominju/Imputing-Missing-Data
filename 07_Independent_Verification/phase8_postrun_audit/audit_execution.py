#!/usr/bin/env python
"""Phase 8 post-run EXECUTION / DATA-INTEGRITY audit (2026-10-31). Imports no project module; read-only on the project.
  E1  task graph rebuilt from protocol v2 (sections 4-7, 11): 1276 expected output markers -> existence, size, mtime window
  E2  scheduler log: every task finished exactly once with rc = 0 (no duplicate completion = no silent overwrite), kills, failures
  E3  failure markers, invalid-run records, leftover temporary files
  E4  data identity / leakage: role-disjoint ids, all trajectories distinct (SHA-256), no exact state shared between test and
      pool / reference / replicate-train / donor trajectories, p = 0.3 and p = 0.7 masks distinct
  E5  execution environment per task (GPU index from training rows, driver epoch from completion time)
usage: python -I audit_execution.py --root <project root> --out <dir>"""
import argparse, glob, hashlib, json, os, re, time
from datetime import datetime
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); A = ap.parse_args()
os.chdir(A.root); OUT = A.out; os.makedirs(OUT, exist_ok=True); P8 = "results/extra/phase8"; LOG = open(f"{OUT}/execution_audit_log.txt", "w")


def say(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); LOG.write(t + "\n"); LOG.flush()


KAPPAS, PS, R = (0.0, 1.0), (0.3, 0.7), 24
ARMS = ["Bfull", "BfullSham1", "BfullSham2", "A1_8", "A2_8", "A1_8d", "A2_8d", "A1_16", "A2_16"]
seed0 = lambda a: 8000 if a.startswith("Bfull") else (8100 if a.endswith(("_8", "_8d")) else 8200)
FREEZE = datetime(2026, 10, 29, 11, 16, 1).timestamp(); ANALYSIS = datetime(2026, 10, 31, 18, 19, 46).timestamp()
DRIVER_CHANGE = datetime(2026, 10, 29, 13, 19, 30).timestamp()      # last pre-installation stop; restart 10-31 16:23

# ---------------- E1 expected outputs
exp = {}
for p in PS:
    for k in KAPPAS:
        for kind in ("raw", "own1", "own2", "imp"):
            exp[f"orc:{kind}:{k}:{p}"] = f"{P8}/oracle/{kind}_k{k}_p{p}.pt"
        exp[f"orc:sat:{k}:{p}"] = f"{P8}/oracle/sat_k{k}_p{p}.json"
        exp[f"knn:eval:{k}:{p}"] = f"{P8}/eval/knn_k{k}_p{p}.csv"; exp[f"knn:d5b:{k}:{p}"] = f"{P8}/eval/d5b_k{k}_p{p}.csv"
        for r in range(R):
            for feat in ("Bfull", "Fact8"):
                exp[f"orc:lat:{k}:{p}:{r}:{feat}"] = f"{P8}/oracle/lat_k{k}_p{p}_r{r}_{feat}.pt"
            exp[f"eval:{k}:{p}:{r}"] = f"{P8}/eval/k{k}_p{p}_r{r}_acc.csv.gz"
            exp[f"p7imp:{k}:{p}:{r}"] = f"{P8}/eval/p7imp_k{k}_p{p}_r{r}.csv.gz"
            for a in ARMS:
                exp[f"train:{k}:{p}:{r}:{a}"] = f"{P8}/runs/rows/k{k}_p{p}_r{r}_{a}_s{seed0(a) + r}.csv"
say(f"== E1 expected task outputs rebuilt from protocol: {len(exp)} (protocol: 1276)")
st = {n: os.stat(f) if os.path.exists(f) else None for n, f in exp.items()}
missing = [n for n, s in st.items() if s is None]; empty = [n for n, s in st.items() if s is not None and s.st_size == 0]
mt = np.array([s.st_mtime for s in st.values() if s is not None])
say(f"   missing {len(missing)}  empty {len(empty)}  mtime range {time.strftime('%F %T', time.localtime(mt.min()))} .. {time.strftime('%F %T', time.localtime(mt.max()))}")
say(f"   all outputs after freeze (10-29 11:16:01): {bool((mt > FREEZE).all())};  all before complete analysis (10-31 18:19:46): {bool((mt < ANALYSIS).all())}")
ck = sorted(glob.glob(f"{P8}/runs/ckpt/*.pt")); say(f"   checkpoints: {len(ck)} (expected 864)")
side = {kind: len(glob.glob(f"{P8}/eval/k*_{kind}.csv.gz")) for kind in ("pair", "imp", "lat")}; say(f"   side files {side} (expected 96 each)")
# ---------------- E2 scheduler log
say("\n== E2 scheduler log")
L = open(f"{P8}/logs/sched8.log").read().splitlines()
starts, ends, kills, other = {}, {}, {}, []
for ln in L:
    m = re.match(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\] (start|end|killed) (\S+)(.*)", ln)
    if not m:
        if "START" in ln or "EXIT" in ln or "analysis" in ln:
            other.append(ln)
        continue
    ts, ev, nm, rest = m.groups(); t = datetime.strptime("2026-" + ts, "%Y-%m-%d %H:%M:%S").timestamp()
    if ev == "start":
        starts.setdefault(nm, []).append(t)
    elif ev == "end":
        rc = int(re.search(r"rc=(-?\d+)", rest).group(1)); ends.setdefault(nm, []).append((t, rc))
    else:
        kills.setdefault(nm, []).append(t)
ok_end = {n: [t for t, rc in v if rc == 0] for n, v in ends.items()}; bad_rc = {n: [rc for t, rc in v if rc != 0] for n, v in ends.items() if any(rc != 0 for t, rc in v)}
dup = {n: v for n, v in ok_end.items() if len(v) > 1}; never = [n for n in exp if n not in ok_end]
say(f"   scheduler sessions: " + " | ".join(o for o in other if "START" in o or "EXIT" in o))
say(f"   tasks with rc=0 completion: {len(ok_end)}  duplicates: {len(dup)}  nonzero rc: {len(bad_rc)} {bad_rc if bad_rc else ''}  expected-but-never-ended: {len(never)}")
restarted = {n: len(v) for n, v in starts.items() if len(v) > 1}
say(f"   tasks started more than once (interrupted by a planned stop, then rerun from scratch): {len(restarted)}")
nonzero_ends_no_rc0 = [n for n in bad_rc if n not in ok_end]
pd.DataFrame([dict(task=n, starts=len(starts.get(n, [])), rc0_ends=len(ok_end.get(n, [])), last_end=time.strftime('%F %T', time.localtime(max(ok_end[n]))) if n in ok_end else "",
                   output=exp.get(n, "")) for n in sorted(set(starts) | set(exp))]).to_csv(f"{OUT}/task_ledger.csv", index=False)
# the marker mtime must be at/after the logged rc=0 end of the same task (no later overwrite by another run)
lag = []
for n, f in exp.items():
    if n in ok_end and st[n] is not None:
        lag.append(st[n].st_mtime - max(ok_end[n]))
lag = np.array(lag); say(f"   marker mtime - logged completion: min {lag.min():.1f}s max {lag.max():.1f}s (markers are written immediately before 'end')")
# ---------------- E3 failure markers / temp files
say("\n== E3 failure markers, invalid runs, temporary files")
fm = glob.glob(f"{P8}/failed/*"); rf = glob.glob(f"{P8}/runs/failed/*"); tmp = [f for f in glob.glob(f"{P8}/**/*", recursive=True) if ".tmp" in os.path.basename(f)]
say(f"   scheduler failure markers {len(fm)}  invalid-AE records {len(rf)}  temp files {len(tmp)} {tmp[:5]}")
# ---------------- E4 data identity / leakage
say("\n== E4 trajectory identity and leakage")
files = sorted(glob.glob(f"{P8}/data/k*_id*.npz")); roles = {}
def role(i):
    if 50000 <= i < 50100: return "rep_train"
    if 50100 <= i < 50200: return "rep_val"
    if 50200 <= i < 51000: return "rep_test"
    if 51000 <= i < 51100: return "donor_train"
    if 51100 <= i < 51200: return "donor_val"
    if 51200 <= i < 52000: return "donor_test"
    if 60000 <= i < 60100: return "pool_train"
    if 60100 <= i < 60200: return "pool_val"
    return "other"
H, X0h, cnt = {}, {}, {}
for f in files:
    k, i = re.search(r"k([\d.]+)_id(\d+)", f).groups(); k, i = float(k), int(i); X = np.load(f)["X"]
    H[(k, i)] = hashlib.sha256(X.tobytes()).hexdigest(); X0h[(k, i)] = X[0].tobytes(); cnt[(k, role(i))] = cnt.get((k, role(i)), 0) + 1
say(f"   trajectories per (kappa, role): {dict(sorted(cnt.items()))}")
exp_cnt = dict(rep_train=24, rep_val=24, rep_test=192, donor_train=24, donor_val=24, donor_test=192, pool_train=40, pool_val=8)
say(f"   counts match protocol 4.1: {all(cnt.get((k, r_), 0) == v for k in KAPPAS for r_, v in exp_cnt.items())}; other ids: {sum(v for (k, r_), v in cnt.items() if r_ == 'other')}")
say(f"   all {len(H)} trajectory arrays distinct (SHA-256): {len(set(H.values())) == len(H)};  all first states distinct: {len(set(X0h.values())) == len(X0h)}")
leak = {}
for k in KAPPAS:
    test_states = set()
    for (kk, i), _ in H.items():
        if kk == k and role(i) == "rep_test":
            test_states.update(map(bytes, np.load(f"{P8}/data/k{k}_id{i}.npz")["X"]))
    for rl in ("pool_train", "pool_val", "rep_train", "donor_test", "donor_train"):
        hit = 0
        for (kk, i), _ in H.items():
            if kk == k and role(i) == rl:
                hit += sum(bytes(s) in test_states for s in np.load(f"{P8}/data/k{k}_id{i}.npz")["X"])
        leak[(k, rl)] = hit
    Sref = np.load(f"{P8}/data/ref_k{k}.npz")["S"]; leak[(k, "kNN_ref_windows")] = int(sum(bytes(s) in test_states for s in Sref.reshape(-1, 6)))
say(f"   exact 6-D states shared with replicate TEST trajectories: {leak}")
say(f"   kNN reference: one time point per independent trajectory (protocol D5a): n_q range " +
    ", ".join(f"k={k}: [{int(np.load(f'{P8}/data/ref_k{k}.npz')['n'].min())}, {int(np.load(f'{P8}/data/ref_k{k}.npz')['n'].max())}] (N={len(np.load(f'{P8}/data/ref_k{k}.npz')['n'])})" for k in KAPPAS))
# ---------------- E5 environment
say("\n== E5 execution environment of the 864 AE trainings")
rows = pd.concat([pd.read_csv(f) for f in glob.glob(f"{P8}/runs/rows/*.csv")], ignore_index=True)
rows["task"] = [f"train:{k}:{p}:{r}:{a}" for k, p, r, a in zip(rows.kappa, rows.p, rows.rep, rows.arm)]
rows["completed"] = rows.task.map(lambda n: max(ok_end[n]) if n in ok_end else np.nan)
rows["driver"] = np.where(rows.completed < DRIVER_CHANGE, "580.105.08", "580.196.07")
tab = rows.groupby(["kappa", "p", "driver"]).size().unstack(fill_value=0); say(tab.to_string())
say("   GPU index (cuda_visible) x driver:\n" + rows.groupby(["driver", "cuda_visible"]).size().unstack(fill_value=0).to_string())
say(f"   code_md5 distinct values: {sorted(rows.code_md5.unique())}")
rows[["exp_id", "kappa", "p", "rep", "arm", "cuda_visible", "driver", "best_val", "epochs_run", "wall_s"]].to_csv(f"{OUT}/training_environment.csv", index=False)
# arm-level comparison of best_val by driver epoch (descriptive; replicate-level randomness, so not a test of driver effects)
say("   mean best_val by (kappa, p, driver) [descriptive]:\n" + rows.groupby(["kappa", "p", "driver"]).best_val.mean().unstack().round(4).to_string())
json.dump(dict(expected=len(exp), missing=len(missing), empty=len(empty), duplicates=len(dup), nonzero_rc=len(bad_rc), never_ended=len(never), restarted=len(restarted),
               failure_markers=len(fm), invalid_runs=len(rf), temp_files=len(tmp), checkpoints=len(ck), side_files=side,
               leakage_exact_state_hits={f"{k}|{r_}": v for (k, r_), v in leak.items()}, all_traj_distinct=len(set(H.values())) == len(H)),
          open(f"{OUT}/execution_audit_summary.json", "w"), indent=1)
say("== DONE")
