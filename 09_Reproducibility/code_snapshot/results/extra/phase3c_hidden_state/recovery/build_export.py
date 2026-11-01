"""Phase 3C export (2026-10-27): no recomputation. Copies original files byte-for-byte into a ZIP, separates the
z-removal bug rows, writes RECOVERY_MANIFEST.md with SHA256 hashes, then re-opens the ZIP and verifies every member.
Only new files are written, all inside recovery/. Run from the project root."""
import hashlib, os, zipfile, datetime
import pandas as pd

O = "results/extra/phase3c_hidden_state"; R = f"{O}/recovery"; N = "results/notes"
ZIP = f"{R}/phase3c_verified_results_20261027.zip"
assert not os.path.exists(ZIP), "refusing to overwrite an existing export"

# derived file: eval_probe.csv minus the 18 bug rows (z-removed label but x actually removed)
P = pd.read_csv(f"{O}/eval_probe.csv")
bug = P.label.str.endswith("_zremoved_refit")
a = P[bug].sort_values(["model", "seed", "probe"]).test_rmse.values
b = P[P.label.str.endswith("_xremoved_refit")].sort_values(["model", "seed", "probe"]).test_rmse.values
assert bug.sum() == 18 and (a == b).all()          # bug rows are bit-identical copies of the x-removed rows
DER = f"{R}/eval_probe_valid_rows.csv"
assert not os.path.exists(DER)
P[~bug].to_csv(DER, index=False)

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def rows(p): return len(pd.read_csv(p)) if p.endswith(".csv") else ""

# (archive path, source path, status)
V = "VALID (original, unchanged)"
items = [(f"csv/{f}", f"{O}/{f}", V) for f in [
    "eval_probe_zremoved.csv", "eval_imputation.csv", "eval_consistency.csv", "eval_dynamics.csv", "eval_blocks.csv",
    "eval_acf.csv", "runs.csv", "observability_local.csv", "observability_global.csv", "fd_baseline.csv"]]
items += [("csv/eval_probe_valid_rows.csv", DER, "VALID (derived: eval_probe.csv minus 18 *_zremoved_refit bug rows)")]
items += [(f"training_logs/{f}", f"{O}/logs/{f}", V) for f in sorted(os.listdir(f"{O}/logs"))]
items += [(f"invalid_or_superseded/{f}", f"{O}/{f}", s) for f, s in [
    ("eval_probe.csv", "MIXED (original, unchanged): 93 valid rows + 18 invalid *_zremoved_refit rows"),
    ("analysis.log", "MIXED (original, unchanged): z-removed lines show the buggy x-removed values")]]
items += [(f"notes/{f}", f"{N}/{f}", "REPORT (original, unchanged)") for f in [
    "phase3c_observability_audit.md", "phase3c_experiment_results.md", "phase3c_state_consistency.md",
    "phase3c_research_decision.md", "phase3c_partial_observation_proposal.md"]]
items += [(f"run_logs/{f}", f"{O}/{f}", "LOG (original, unchanged)") for f in [
    "tests.log", "tests_leakage_rerun.log", "lane_gpu0.log", "lane_gpu1.log", "evaluate.log", "multiview_fix.log",
    "observability_audit.log"]]
items += [(f"run_logs/test_artifacts/{f}", f"{O}/test_artifacts/{f}", "TEST ARTIFACT (not a result)")
          for f in ["leaktest.csv", "logs/leaktest.csv"]]
items += [(f"code/{f}", f"{O}/{f}", "CODE (original, unchanged)") for f in [
    "train.py", "run_lane.sh", "evaluate.py", "multiview_fix.py", "analyze.py", "observability_audit.py"]]
items += [(f"code/experiments_fdr/{f}", f"experiments/fdr/{f}", "CODE (original, unchanged)") for f in [
    "xz_data.py", "probes.py", "features.py", "model.py", "loss.py", "impute.py", "diag.py",
    "tests/test_phase3c_leakage.py", "tests/test_fdr.py"]]
items += [("code/recovery_build_export.py", f"{R}/build_export.py", "CODE (this export script)")]

ck = sorted(os.listdir(f"{O}/checkpoints"))
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
L = [f"# Phase 3C — RECOVERY_MANIFEST ({now} KST)", "",
     "## 결론", "",
     "- 서버에서 **누락되거나 손상된 Phase 3C CSV는 없다** (Case A). 따라서 재학습도, 평가 재계산도 하지 않았다.",
     "- 이름에 `recovered`가 붙은 파일은 없다. 실제 복구가 일어나지 않았기 때문이다.",
     "- ZIP의 모든 원본 파일은 서버 원본을 바이트 그대로 복사했다. 아래 SHA256은 서버 원본의 해시다.",
     "- 새로 만든 파일은 두 개뿐이다.",
     "  - `eval_probe_valid_rows.csv`: 원본 `eval_probe.csv`에서 버그 행 18개만 제외한 파생 파일. 값은 바꾸지 않았다.",
     "  - 이 매니페스트.",
     "",
     "## z-removal 버그",
     "",
     "- 첫 `evaluate.py`는 \"z 제거\" 입력을 만들 때 x 열을 지웠다 (`Mzr[:, 0] = 0`).",
     "  - 그래서 `eval_probe.csv`의 `latent_{C,FDR}-xz_zremoved_refit` 18행(2 모델 × 3 seed × 3 probe)은 `*_xremoved_refit` 행과 비트 단위로 같다. 이번 감사에서 확인했다.",
     "  - 이 행들은 **무효**다.",
     "- 수정 코드:",
     "  - `evaluate.py` 79행이 현재 `Mzr[:, 1] = 0`로 고쳐져 있다. 이 수정은 CSV 생성(12:41) 이후인 12:42에 이뤄졌다.",
     "  - 따라서 원본 `eval_probe.csv`는 수정 전 코드로 만들어졌다.",
     "- 올바른 z 제거 결과는 `eval_probe_zremoved.csv`다.",
     "  - 행 수: 2 모델 × 3 seed × 3 probe × 2 mode = 36행.",
     "  - mode는 `refit_on_z_removed`와 `natural_probe_applied_to_z_removed` 두 가지다.",
     "  - 생성 코드는 `multiview_fix.py`다. 이 코드는 `assert`로 x 마스크가 그대로이고 z 마스크가 0임을 확인한다.",
     "- 보고서 `phase3c_experiment_results.md` §5의 z 제거 수치는 `eval_probe_zremoved.csv`와 일치한다. 즉 보고서는 버그 행을 인용하지 않는다.",
     "  - 재적합 MLP: C 0.106/0.114/0.125, FDR 0.085/0.090/0.109.",
     "  - 전이: C 0.172/0.192/0.184, FDR 0.163/0.153/0.159.",
     "- `analysis.log`의 z-removed 줄(22–37행 블록과 81–86행 블록의 `zremoved`)도 같은 버그 값이다.",
     "",
     "## 학습 8회 대조 (runs.csv ↔ 체크포인트 ↔ 학습 로그)",
     "",
     "8개 체크포인트 모두 CPU에서 로드된다. 각 run은 다음이 서로 일치한다.",
     "",
     "- 파라미터 수가 `runs.csv`의 `n_params`와 같다.",
     "- 학습 로그 행 수가 `epochs_run`과 같다.",
     "- 로그의 val 최소 epoch이 `best_epoch`와 같다.",
     "",
     "모든 run은 다음 조건을 만족한다.",
     "",
     "- 종료 사유가 `patience`다.",
     "- `hit_cap = False`다.",
     "- `nonfinite_steps = 0`이다.",
     "- `code_md5`가 f85b14f9다.",
     "",
     "`eval_imputation.csv`의 `recorded`는 `runs.csv`의 `test_rmse`와 같다.",
     "",
     "| run | 체크포인트 SHA256 (앞 16자) |", "|---|---|"]
L += [f"| {c} | {sha(f'{O}/checkpoints/{c}')[:16]} |" for c in ck]
L += ["", "체크포인트와 `*_xhat_past.npy`는 ZIP에 넣지 않았다. 서버 원본 경로는 다음과 같다.", "",
      f"- `{os.path.abspath(O)}/checkpoints/`", f"- `{os.path.abspath(O)}/`", "",
      "## 의도된 NaN (손상 아님)", "",
      "| 파일 | 열 | NaN 행 수 | 의도된 이유 |",
      "|---|---|---|---|",
      "| `eval_probe.csv` | `coverage` | 108 | FD 기준선 행에만 정의된다 |",
      "| `eval_probe.csv` | `val_rmse` | 6 | 평균 예측기와 FD 해석식에는 검증 선택이 없다 |",
      "| `eval_imputation.csv` | `rmse_z` | 2 | x-only 모델에는 z가 없다 |",
      "| `runs.csv` | `window`, `direction` | 각 8 | cfg에 값이 없다(None) |",
      "| `observability_local.csv` | `cond`, `amp_y` | 각 2087 | 계수 결손(rank < 3) 패턴에서는 정의되지 않는다. 주로 lag-0만 있는 패턴이다 |",
      "",
      "그 밖의 CSV는 다음을 확인했다.",
      "",
      "- 모든 행의 필드 수가 헤더와 같다.",
      "- 중복 행이 없다.",
      "- 파일 끝이 줄바꿈으로 끝난다(잘린 마지막 행 없음).",
      "",
      "## 파일 목록",
      "",
      "| ZIP 경로 | 서버 원본 | 행 수 | SHA256 | 상태 |",
      "|---|---|---|---|---|"]
L += [f"| `{a_}` | `{s}` | {rows(s)} | `{sha(s)}` | {st} |" for a_, s, st in items]
L += ["", "## 검증 상태", "",
      "- [Verified] 파일 존재, 구조, 해시, 체크포인트 로드와 메타데이터 대조, 버그 행의 동일성.",
      "- [Verified] 보고서의 주요 수치와 CSV가 일치한다.",
      "  - 대치 RMSE.",
      "  - MLP 평균: FDR 0.0415, C 0.0557.",
      "  - 원본 창: xz 0.079, x 0.109.",
      "  - x-only 모델: C-x 0.169, FDR-x 0.086.",
      "  - z 제거 수치.",
      "- 재계산 비교는 하지 않았다. 원본이 남아 있어 재계산할 이유가 없었다(Case A).", ""]
open(f"{R}/RECOVERY_MANIFEST.md", "w").write("\n".join(L))
items.append(("RECOVERY_MANIFEST.md", f"{R}/RECOVERY_MANIFEST.md", "MANIFEST"))

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for a_, s, _ in items: z.write(s, f"phase3c_verified_results_20261027/{a_}")
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    names = z.namelist()
    assert len(names) == len(items)
    for a_, s, _ in items:
        assert hashlib.sha256(z.read(f"phase3c_verified_results_20261027/{a_}")).hexdigest() == sha(s), a_
open(ZIP + ".sha256", "w").write(f"{sha(ZIP)}  {os.path.basename(ZIP)}\n")
print("OK", len(names), "members,", os.path.getsize(ZIP), "bytes")
for n in names: print(" ", n)
