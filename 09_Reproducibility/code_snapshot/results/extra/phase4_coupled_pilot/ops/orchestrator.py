"""Session-independent pilot orchestrator (CPU only; launched with setsid nohup; never starts or stops training).
Expected runs are a FIXED manifest (Track A 25, Track B 48 = exactly the jobs launched at 16:12 / 16:18 KST). A run's state is
decided from evidence, not from process search:
  completed : lane log 'end <job> rc=0' AND rows/<exp>.csv AND checkpoints/<exp>.pt AND finite=True, nonfinite_steps=0,
              all checkpoint tensors finite
  failed    : 'end <job> rc!=0', or rc=0 with a missing/non-finite artefact
  skipped   : 'skip <job>' (reservation guard)
  pending   : none of the above (still queued or running)
A track is terminal when no run is pending, or when no lane process of that track exists any more (pending -> 'lost').
Lane liveness = /proc/<pid>/cmdline containing 'run_lane.sh' and the track's train module; this script's own command
line contains neither, so it cannot match itself. When a track is terminal: its CPU evaluation (idempotent; completed runs
only) then its analysis; status written to PILOT_STATUS.md / pilot_status.csv after every step."""
import datetime, glob, os, re, subprocess, time
import numpy as np, pandas as pd, torch

ROOT = "/home/minju/faicpaston/Imputing_Missing_Data"; os.chdir(ROOT)
PY = os.path.expanduser("~/.conda/envs/reef_gpu/bin/python")
OPS = "results/extra/phase4_coupled_pilot/ops"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=".pydeps:.", OMP_NUM_THREADS="3")


def now(): return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg):
    with open(f"{OPS}/orchestrator.log", "a") as f:
        f.write(f"[{now()}] {msg}\n")


A_JOBS = [("FDR1/8", s) for s in (0, 3, 4)] + [(f"FDR{a}", s) for a in ("0.35", "0.65", "1.0") for s in range(5)] + \
         [("C", s) for s in (3, 4)] + [("Ccap", s) for s in range(5)]
B_JOBS = [(mo, k, p, s) for s in (0, 1, 2) for k in ("1.0", "0.0") for p in ("0.3", "0.7") for mo in ("C-Scalar", "FDR-Scalar", "C-Learned", "FDR-Learned")]
TRACKS = {
    "A": dict(dir="results/extra/phase3d2_lag_weight", module="results.extra.phase3d2_lag_weight.train",
              jobs=[(f"--variant {v} --seed {s}", f"p3d2_{v.replace('/', 'o')}_s{s}") for v, s in A_JOBS],
              evaluate=[PY, "results/extra/phase3d2_lag_weight/evaluate.py"], eval_out="results/extra/phase3d2_lag_weight/eval_trackA.csv",
              analyze=[PY, "results/extra/phase3d2_lag_weight/analyze.py"], analysis_log="results/extra/phase3d2_lag_weight/analysis.log"),
    "B": dict(dir="results/extra/phase4_coupled_pilot", module="results.extra.phase4_coupled_pilot.train",
              jobs=[(f"--model {mo} --kappa {k} --p {p} --seed {s}", f"p4_k{k}_t0_p{p}_{mo}_s{s}") for mo, k, p, s in B_JOBS],
              evaluate=[PY, "results/extra/phase4_coupled_pilot/evaluate.py", "p4_k*_t0_*", "pilot"], eval_out="results/extra/phase4_coupled_pilot/eval_models_pilot.csv",
              analyze=[PY, "results/extra/phase4_coupled_pilot/analyze.py", "pilot"], analysis_log="results/extra/phase4_coupled_pilot/analysis_pilot.log"),
}


def lanes_alive(module):
    n = 0
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            c = open(p, "rb").read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "run_lane.sh" in c and module in c:
            n += 1
    return n


def status(tr):
    T = TRACKS[tr]; txt = "\n".join(open(f).read() for f in glob.glob(f"{T['dir']}/lanes/lane*.log"))
    ends = {m.group(1).strip(): int(m.group(2)) for m in re.finditer(r"\] end (.*?) rc=(\d+)", txt)}
    skips = {m.group(1).strip() for m in re.finditer(r"\] skip (.*?): near", txt)}
    rows = []
    for job, eid in T["jobs"]:
        rowf, ck = f"{T['dir']}/rows/{eid}.csv", f"{T['dir']}/checkpoints/{eid}.pt"; note = ""
        if job in ends:
            st = "completed" if ends[job] == 0 else "failed"
            if st == "completed":
                if not (os.path.exists(rowf) and os.path.exists(ck)):
                    st, note = "failed", "rc=0 but rows/checkpoint missing"
                else:
                    r = pd.read_csv(rowf).iloc[0]
                    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
                    fin = all(bool(torch.isfinite(v).all()) for v in sd.values())
                    if not (bool(r.finite) and int(r.nonfinite_steps) == 0 and fin):
                        st, note = "failed", "non-finite output/steps/parameters"
                    else:
                        note = f"test_rmse={r.test_rmse:.4f} epochs={int(r.epochs_run)} best={int(r.best_epoch)} stop={r.stop_reason}"
        elif job in skips:
            st = "skipped"
        else:
            st = "pending"
        rows.append(dict(track=tr, exp_id=eid, job=job, status=st, rc=ends.get(job, ""), note=note))
    return pd.DataFrame(rows)


def write_status(S, steps):
    S.to_csv(f"{OPS}/pilot_status.csv", index=False)
    with open(f"{OPS}/PILOT_STATUS.md", "w") as f:
        f.write(f"# Pilot status (updated {now()} KST by ops/orchestrator.py)\n\n")
        for tr, g in S.groupby("track"):
            f.write(f"- Track {tr}: " + ", ".join(f"{k} {v}" for k, v in g.status.value_counts().items()) + f" / expected {len(g)}\n")
        f.write("\n## Steps\n\n" + "\n".join(f"- {k}: {v}" for k, v in steps.items()) + "\n\n## Runs\n\n| track | exp_id | status | rc | note |\n|---|---|---|---|---|\n")
        for _, r in S.iterrows():
            f.write(f"| {r.track} | {r.exp_id} | {r.status} | {r.rc} | {r.note} |\n")


def run_step(name, cmd, steps):
    log(f"start {name}: {' '.join(cmd)}"); steps[name] = f"running since {now()}"
    with open(f"{OPS}/{name}.log", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT)
    steps[name] = f"rc={rc} at {now()}"; log(f"end {name} rc={rc}"); return rc


def main():
    log(f"orchestrator start pid={os.getpid()}"); steps = {}; done = {"A": False, "B": False}
    while not all(done.values()):
        S = pd.concat([status("A"), status("B")]); write_status(S, steps)
        for tr in ("A", "B"):
            if done[tr]:
                continue
            g = S[S.track == tr]; pend = (g.status == "pending").sum(); alive = lanes_alive(TRACKS[tr]["module"])
            if pend and alive:
                continue
            if pend and not alive:
                S.loc[(S.track == tr) & (S.status == "pending"), "status"] = "lost"; log(f"track {tr}: {pend} pending runs but no lane alive -> lost")
            log(f"track {tr} terminal: {dict(g.status.value_counts())}")
            T = TRACKS[tr]
            for step in ("evaluate", "analyze"):                     # idempotent: a .done marker is written only on rc=0
                mark = f"{OPS}/{step}_{tr}.done"
                if os.path.exists(mark):
                    steps[f"{step}_{tr}"] = "already done (marker)"; continue
                if run_step(f"{step}_{tr}", T[step], steps) != 0:
                    break
                open(mark, "w").write(now())
            done[tr] = True; write_status(pd.concat([status("A"), status("B")]), steps)
        time.sleep(30)
    log("orchestrator finished")


if __name__ == "__main__":
    main()
