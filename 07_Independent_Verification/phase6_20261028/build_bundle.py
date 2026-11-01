"""Build Phase 6 review ZIPs (read-only on project files; copies only). Run from project root:
  python results/review_packages/phase6_20261028/build_bundle.py results/review_packages/phase6_20261028
No new conclusions are written; FILE_INDEX.md only points to existing files and sections."""
import datetime, glob, hashlib, json, os, sys, zipfile

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST"); ROOT = os.getcwd()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mt(p):
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")


# ======================= ZIP 1: documents =======================
DOCS = [("results/notes/phase6_mathematical_errata.md", "G0 정오표: E1–E9, 명제 E3.1(κ=0 Real–Sham 교환가능성), Sham donor 감사(E6), MV-shift(E7)"),
        ("results/notes/phase6_g1_protocol.md", "G1 사전 등록(15:08 고정): 명제 G1.1–G1.6(대칭, L0의 E[y|U]=0, 층 분해, 인증 부등식, witness), 추론 단위"),
        ("results/notes/phase6_g1_results.md", "G1 결과: L0 인증량·LCB, L0c0/L0c1, witness, 구조 감사"),
        ("results/notes/phase6_g2_g3_protocol.md", "G2/G3 사전 등록(15:04 고정), §9 변경 기록, §10 S1R 추가 사전 등록(20:10)"),
        ("results/notes/phase6_g2_g3_results.md", "G2/G3 결과, κ=0 음성 대조 실패, 사후 진단 J/S(§5), S1R(§6)"),
        ("results/notes/phase6_final_report.md", "Phase 6 최종 보고 Q1–Q6, 사전 등록/사후 구분표"),
        ("research_handover_latest.md", "인수인계 진입점 (패키지 생성 직전 버전; 생성 기록 추가 전)")]
with zipfile.ZipFile(f"{OUT}/phase6_review_documents.zip", "w", zipfile.ZIP_DEFLATED) as z:
    idx = [f"# FILE_INDEX — phase6_review_documents.zip\n",
           f"생성: {now}. 원본 파일을 수정하지 않고 복사만 했다. 프로젝트 루트: `{ROOT}`. ZIP 내 경로 = 프로젝트 루트 기준 실제 상대 경로.\n",
           "| ZIP 내 경로 | 역할 | 최종 수정 | 크기(B) | SHA-256 |", "|---|---|---|---|---|"]
    for p, role in DOCS:
        z.write(p, p); idx.append(f"| `{p}` | {role} | {mt(p)} | {os.path.getsize(p)} | `{sha(p)}` |")
    idx.append("\n동반 자료: `phase6_verification_bundle.zip` (코드, 원본 평가 CSV, 분석 CSV, 고정 기록). 근거 위치 안내는 그 ZIP의 FILE_INDEX.md §A에 있다.")
    z.writestr("FILE_INDEX.md", "\n".join(idx) + "\n")

# ======================= ZIP 2: verification bundle =======================
E = "results/extra/phase6"; G = []                       # (path, group, role)


def add(pattern, group, role):
    ps = sorted(p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p) and "__pycache__" not in p)
    assert ps, f"no files for {pattern}"
    for p in ps:
        G.append((p, group, role))


# --- code ---
add("experiments/phase6/*.py", "code", "Phase 6 코드(G0 점검, 자료 생성, Arm/donor 조립, 학습, 평가, G1, 분석)")
add("experiments/phase6/*.sh", "code", "G2 학습 lane 스크립트(마감 guard, rc 기록)")
add("experiments/phase6/tests/*.py", "test", "Phase 6 무결성 테스트 9개")
add("experiments/phase6/diag/*.py", "code-posthoc", "사후 진단 J/S: 자료 생성(결합 대 분리 적분), 학습, 평가")
add("experiments/phase6/diag/*.sh", "code-posthoc", "사후 진단 lane·평가 루프")
add("experiments/phase6/s1r/*.py", "code-S1R", "S1R(사전 등록 재현): 자료 생성, 학습, 평가, 분석")
add("experiments/phase6/s1r/*.sh", "code-S1R", "S1R lane·평가 루프")
for p, role in [("experiments/phase5/core.py", "의존: ArmModel(FDR-Learned 구조), encode_series(past-only), raw_features, mcar, seed_of"),
                ("experiments/phase5/train.py", "의존: train6.py가 파생된 원본 학습 루프(차이는 train6 docstring 참조)"),
                ("experiments/phase5/evaluate.py", "의존: ridge_probe"),
                ("experiments/phase5/obs_audit.py", "의존: f6, j6, back_jac (Oracle Jacobian)"),
                ("experiments/phase5/external.py", "의존: G0 MV-shift 점검 대상(shift_index, setup)"),
                ("experiments/phase5/tests/test_phase5.py", "Phase 5 무결성 테스트(8)"),
                ("experiments/phase5/tests/test_external.py", "Gate F 무결성 테스트(3)"),
                ("experiments/fdr/loss.py", "학습 Target: 다중도 보정 FDR 손실(목표 열의 숨긴 항목)"),
                ("src/models.py", "의존: group_mlp, mlp(구조 구성 요소)"),
                ("src/delay_embedding.py", "의존: delay_embed, pad_front(마스킹 후 창 절단)"),
                ("src/lorenz.py", "의존: coupled_lorenz_rhs, generate_coupled_lorenz, integrate, lorenz_rhs"),
                ("src/missingness.py", "참고: Phase 5 마스크 유틸리티(MF §1.3에서 인용)")]:
    if os.path.exists(p):
        G.append((p, "code-dep", role))
# --- reference documents (prior phase, cited by the proofs) ---
for p, role in [("results/notes/phase5_mathematical_foundations.md", "참조: 보조정리 S, 따름정리 1–3(G1.2 증명이 인용), 명제 2·3·5·6"),
                ("results/notes/phase5_hypothesis_protocol.md", "참조: Phase 5 사전 등록(정오표 E3·E5·E8의 대상)"),
                ("results/notes/phase5_observability_information_audit.md", "참조: Gate C(국소 Jacobian, 대칭 수치 확인)")]:
    G.append((p, "ref-doc", role))
# --- freeze / change records ---
add(f"{E}/PROTOCOL_FREEZE_g1.txt", "freeze", "G1 프로토콜·코드 고정 시각과 md5, 분석 스크립트 고정(결과 전)")
add(f"{E}/PROTOCOL_FREEZE_g2g3.txt", "freeze", "G2/G3 프로토콜·코드 고정 시각과 md5, 분석 스크립트 고정(결과 전)")
add(f"{E}/diag_k0/FREEZE.txt", "freeze", "사후 진단 코드 고정(진단 결과 전)")
add(f"{E}/s1r/FREEZE.txt", "freeze", "S1R 사전 등록 고정(20:10) + SyntaxError 수정 기록")
# --- G0 ---
add(f"{E}/g0/*.csv", "G0", "G0 점검 결과: MV-shift 시간 오프셋, Phase 5 Sham donor 지도, B−Sham 의존성 감도")
# --- G1 ---
add(f"{E}/g1/k*_p*.csv", "G1-raw", "G1 원본 평가: test 궤적(32)×마스크(10)×층×양(risk, cert, T, Dq, sign err)의 합과 n")
add(f"{E}/g1/k*_train.json", "G1-raw", "G1 학습 로그(적합별 best_val, epoch)")
add(f"{E}/g1/log_*.txt", "G1-raw", "G1 실행 로그")
add(f"{E}/g1/analysis_g1.csv", "G1-analysis", "G1 분석: 비율 추정량, cluster bootstrap(궤적 단위) 95% CI / 단측 LCB")
# --- G2/G3 ---
add(f"{E}/eval/k*_g2.csv", "G2-raw", "G2 원본 평가: 복제본별 Arm×j×probe×test 궤적×층 SSE/n (κ=0 음성 대조 포함)")
add(f"{E}/eval/k*_g3.csv", "G3-raw", "G3 원본 평가: 공유 6-D readout, 마스크 5개, 마스크-only/Oracle σ_min 층, 좌표별 SSE/n, 부호 오류")
add(f"{E}/eval/k*_g3pair.csv", "G3-raw", "G3 원본 평가: 같은 상태의 마스크 쌍 출력 차이(안정성)")
add(f"{E}/eval/scheduler.log", "G2-run", "평가 스케줄러 로그")
add(f"{E}/eval/log_*.txt", "G2-run", "평가별 실행 로그")
add(f"{E}/analysis/*.csv", "G2G3-analysis", "사전 등록 분석 출력: 복제본 지표, 수준, 대비, DiD, 고정 순서, G3 정확도·안정성·부호·기준")
add(f"{E}/runs/rows/*.csv", "G2-run", "학습 960개 행: n_params, best_val, epoch, cap, nonfinite, code_md5, cfg, mask_seeds")
add(f"{E}/runs/logs/*.csv", "G2-run", "학습 960개 epoch별 train loss / val 기준")
add(f"{E}/runs/lanes/*", "G2-run", "작업 목록과 lane 로그(rc)")
add(f"{E}/jac_run.log", "G3-run", "Oracle Jacobian 계산 로그")
add(f"{E}/data/gen.log", "data", "Phase 6 궤적 생성 로그")
# --- post-hoc diagnostic ---
add(f"{E}/diag_k0/diag_summary.csv", "diag", "사후 진단 요약(J, S, J−S)")
add(f"{E}/diag_k0/eval_J/*.csv", "diag", "사후 진단 원본 평가(J)")
add(f"{E}/diag_k0/eval_S/*.csv", "diag", "사후 진단 원본 평가(S)")
add(f"{E}/diag_k0/runs_J/rows/*.csv", "diag", "진단 학습 행(J)")
add(f"{E}/diag_k0/runs_S/rows/*.csv", "diag", "진단 학습 행(S)")
add(f"{E}/diag_k0/runs_J/logs/*.csv", "diag", "진단 학습 로그(J)")
add(f"{E}/diag_k0/runs_S/logs/*.csv", "diag", "진단 학습 로그(S)")
add(f"{E}/diag_k0/lanes/*", "diag", "진단 작업 목록·lane 로그")
add(f"{E}/diag_k0/eval_loop.log", "diag", "진단 평가 루프 로그")
# --- S1R ---
add(f"{E}/s1r/s1r_results.csv", "S1R", "S1R 사전 등록 분석 결과")
add(f"{E}/s1r/meta_g2S1_plus_s1r.csv", "S1R", "보조(탐색) 메타: G2 S1 + S1R")
add(f"{E}/s1r/eval/*.csv", "S1R", "S1R 원본 평가")
add(f"{E}/s1r/runs/rows/*.csv", "S1R", "S1R 학습 행")
add(f"{E}/s1r/runs/logs/*.csv", "S1R", "S1R 학습 로그")
add(f"{E}/s1r/lanes/**/*", "S1R", "S1R 작업 목록·lane 로그(첫 SyntaxError 실행 로그 failed_syntax_2009 포함)")
add(f"{E}/s1r/eval_loop.log", "S1R", "S1R 평가 루프 로그")

# --- excluded large files: hashes only ---
EXCL = []
for pat, why in [(f"{E}/data/*.npz", "G2/G1 궤적(1,256)"), (f"{E}/diag_k0/data_J/*.npz", "진단 궤적 J"), (f"{E}/diag_k0/data_S/*.npz", "진단 궤적 S"),
                 (f"{E}/s1r/data/*.npz", "S1R 궤적"), (f"{E}/runs/ckpt/*.pt", "G2 체크포인트"), (f"{E}/diag_k0/runs_J/ckpt/*.pt", "진단 체크포인트 J"),
                 (f"{E}/diag_k0/runs_S/ckpt/*.pt", "진단 체크포인트 S"), (f"{E}/s1r/runs/ckpt/*.pt", "S1R 체크포인트"), (f"{E}/jac/*.npz", "Oracle Jacobian (jac6.py로 재생성 가능)")]:
    for p in sorted(glob.glob(pat)):
        EXCL.append(dict(path=p, kind=why, bytes=os.path.getsize(p), sha256=sha(p)))

manifest = dict(created=now, project_root=ROOT, n_files=len(G),
                files=[dict(path=p, group=g, role=r, bytes=os.path.getsize(p), mtime=mt(p), sha256=sha(p)) for p, g, r in G],
                excluded_hash_only=EXCL)
excl_summary = {}
for e in EXCL:
    s = excl_summary.setdefault(e["kind"], [0, 0]); s[0] += 1; s[1] += e["bytes"]

GUIDE = f"""# FILE_INDEX — phase6_verification_bundle.zip

생성: {now}. 프로젝트 루트 `{ROOT}`. ZIP 내 경로 = 실제 상대 경로. 원본은 수정하지 않고 복사만 했다.
- 파일별 SHA-256·크기·mtime·역할은 `MANIFEST.json`(files)에 있다.
- 제외한 대용량 파일(궤적 npz, 체크포인트, Oracle Jacobian)은 SHA-256만 `MANIFEST.json`(excluded_hash_only)에 기록했다.
- 이 색인은 위치만 안내한다. 새 결론은 포함하지 않는다.
- 문서 7개는 동반 ZIP `phase6_review_documents.zip`에 있다.

## A. 검토 항목별 근거 위치

**1. L0 조건에서 E[y_j|U_j] = 0의 증명**
- `phase6_g1_protocol.md` §2: 명제 G1.1(생성기 법칙의 Γ×Γ, P 불변), 따름정리 G1.2.
- 인용하는 보조정리 S와 따름정리 3 논증: `results/notes/phase5_mathematical_foundations.md` §2.3.
- 생성기 초기분포: `experiments/phase6/gen6.py:s0_of`. 하위계별 독립 U(−15, 15), U(5, 40).
- 대칭 수치 확인: Phase 5 Gate C `symmetry_check.csv`(Phase 5 검증 ZIP).
- L0 층의 정의(마스크만): `experiments/phase6/g1_info.py`의 `Lc == 0`(자기 x 행 마스크 합).

**2. Δ*의 양의 통계적 하한과 독립 표본 단위**
- 명제 G1.3–G1.4: `phase6_g1_protocol.md` §2. 추론: §4.
- 계산 코드: `experiments/phase6/g1_info.py`
  - `cert_<kind>` = y² − (y − f̄)²
  - f̄ = fold A/B 평균, test와 독립
- 집계: `experiments/phase6/analyze_g1.py`
  - 궤적 단위 cluster bootstrap 10,000회
  - 단측 5% 분위 = LCB
  - overall_lower = p⁸ · LCB
- 원본 합: `{E}/g1/k1.0_p0.7.csv`(stratum = L0, quantity = cert_UV / y2 / risk_UV_avg).
  - 단위는 test 궤적 7200–7231(32개 독립)이고, 궤적당 마스크 10개.
- 결과: `{E}/g1/analysis_g1.csv`(stratum = L0, quantity = G(UV), bound = lcb).

**3. L0c0에서 현재값이 아닌 지연 정보가 기여한다는 근거**
- 층 정의: `g1_info.py`의 `L0c0 = (Lc == 0) & (cur == 0)`. cur = 상대 x lag-0 마스크.
- 입력 집합: `feats()`
  - UVc = 자기 창 + 상대 lag 0만
  - UV = 자기 창 + 상대 8-lag 창
- 결과: `analysis_g1.csv`(stratum ∈ {{L0c0, L0c1}}, quantity ∈ {{G(UV), G(UVc)}}).
- 서술: `phase6_g1_results.md` §1.1, `phase6_final_report.md` Q3.
- c0/c1이 상대 마스크의 함수라는 점과 그에 따른 논증: `phase6_g1_results.md` §1.1 첫 문단.

**4. S1 음성 대조 실패와 S1R 재설계의 전체 기록**
- 사전 규칙: `phase6_g2_g3_protocol.md` §5(음성 대조), §7(보류 규칙).
- 결과: `{E}/analysis/g2_contrasts.csv`(kappa = 0, p = 0.7, contrast = 'B - Sham'), `{E}/analysis/g2_fixed_sequence.csv`, `{E}/analysis/g2_replicate_metric.csv`.
- 원본: `{E}/eval/k0.0_p0.7_r*_g2.csv`(24개).
- 사후 진단:
  - 코드: `experiments/phase6/diag/*`. 고정: `{E}/diag_k0/FREEZE.txt`.
  - 결과: `{E}/diag_k0/diag_summary.csv`, `eval_J/`, `eval_S/`.
  - 서술: `phase6_g2_g3_results.md` §5, §5.1.
- S1R:
  - 사전 등록: `phase6_g2_g3_protocol.md` §9–§10. 고정과 수정 기록: `{E}/s1r/FREEZE.txt`.
  - 코드: `experiments/phase6/s1r/*`.
  - 원본: `{E}/s1r/eval/`. 결과: `s1r_results.csv`.
  - 첫 실행 실패 로그: `{E}/s1r/lanes/failed_syntax_2009/`.
  - 서술: `phase6_g2_g3_results.md` §6.

**5. Sham donor의 독립성과 κ = 0 음성 대조의 정확한 가정**
- 가정과 증명: `phase6_mathematical_errata.md` E3(명제 E3.1, 가정 (i)–(iv)), E4, E6.
- Phase 5 donor 지도: `{E}/g0/phase5_sham_donor_map.csv`. 의존성 감도: `{E}/g0/phase5_bsham_dependence.csv`.
- Phase 6 donor 선택: `experiments/phase6/core6.py:ids`, `build`
  - 복제본마다 train/val/test donor가 독립
  - 고정 정규화 하나
- 진단과 S1R의 donor: `diag/k0_common.py:ids`, `s1r/common.py:ids`.
- 테스트: `experiments/phase6/tests/test_phase6.py`
  - `t_roles_never_reused_across_replicates`
  - `t_masks_identical_across_arms_and_sham_columns`
  - `t_seeds_disjoint_from_phase5_and_unique`

**6. H3 6차원 공유 상태 복원 실패의 정확한 실험 조건**
- 사양: `phase6_g2_g3_protocol.md` §6(readout, 특징 집합, 층, 사전 기준 (a)–(c)).
- 코드: `experiments/phase6/evaluate6.py`
  - G3_FEATS
  - `mlp_fit(..., width = 128)`, 6-D target = 정규화 상태
  - 마스크 e = 0..4
  - `joint_lab`, `sig_min`
- Oracle Jacobian: `experiments/phase6/jac6.py`.
- AE 학습 목표(자기 view만 복원): `core6.py:arm_spec`의 outs, `experiments/fdr/loss.py`, `train6.py`(손실은 목표 열에만).
- 결과: `{E}/analysis/g3_criteria.csv`(approx_state_recovery), `g3_accuracy_per_rep.csv`, `g3_stability_per_rep.csv`, `g3_signerr_per_rep.csv`.
- 서술: `phase6_g2_g3_results.md` §3–§4.

**7. 사전 등록 분석과 사후 탐색 분석의 구분**
- 표: `phase6_final_report.md` "사전 등록 결과와 사후 탐색의 구분".
- 고정 시각과 해시:
  - `{E}/PROTOCOL_FREEZE_g1.txt`
  - `{E}/PROTOCOL_FREEZE_g2g3.txt`(분석 스크립트 고정 줄 포함)
  - `{E}/diag_k0/FREEZE.txt`(사후)
  - `{E}/s1r/FREEZE.txt`(S1R)
- 사후 코드는 `experiments/phase6/diag/`에 분리했다. 사후 결과는 `{E}/diag_k0/`, 메타는 `{E}/s1r/meta_g2S1_plus_s1r.csv`에 있다.

## B. 실험별 설정 Manifest (요약; 정확한 값은 코드와 rows CSV의 cfg / mask_seeds 열)

| 실험 | 코드 | 설정 | 출력 |
|---|---|---|---|
| G0 | `g0_checks.py` | CPU, Phase 5 자료 읽기 전용 | `g0/` |
| G1 | `g1_info.py` | κ ∈ {{0, 1}} × p ∈ {{0, 0.3, 0.7}}; MLP 3 × 256, fold A/B 각 16 궤적, val 8, test 32 × 마스크 10 | `g1/` |
| G2/G3 | `gen6.py`, `core6.py`, `train6.py`, `evaluate6.py` | κ 2 × p 2 × 복제본 24 × Arm 10 = 960; CFG = Phase 5와 같음(rows `cfg` 열); init seed 6000 + r | `eval/`, `analysis/`, `runs/` |
| 사후 진단 | `diag/` | κ = 0, p = 0.7, 복제본 24 × {{J, S}} × {{B1, B2, Sham1, Sham2}} = 192 | `diag_k0/` |
| S1R | `s1r/` | κ ∈ {{1, 0}}, p = 0.7, 복제본 24 × {{B1, B2, Sham1, Sham2}} = 192 | `s1r/` |

**제외 파일 (SHA-256은 MANIFEST.json)**
""" + "\n".join(f"- {k}: {v[0]}개, {v[1] / 1e6:.1f} MB" for k, v in excl_summary.items()) + """

## C. 무결성 테스트
- `experiments/phase6/tests/test_phase6.py`: 9개
- `experiments/phase5/tests/test_phase5.py`: 8개
- `experiments/phase5/tests/test_external.py`: 3개
- 패키지 생성 시 원 프로젝트에서 재실행한 결과는 아래 §E에 적는다.

## D. 파일 목록 (그룹별 개수)
""" + "\n".join(f"- {g}: {sum(1 for _, gg, _ in G if gg == g)}개" for g in dict.fromkeys(g for _, g, _ in G)) + "\n"

with zipfile.ZipFile(f"{OUT}/phase6_verification_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p, _, _ in G:
        z.write(p, p)
    z.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    z.writestr("FILE_INDEX.md", GUIDE + "\n## E. 패키지 생성 시 점검\n" + (open(f"{OUT}/_checks.md").read() if os.path.exists(f"{OUT}/_checks.md") else "(아래 SHA256SUMS와 별도 점검 기록 참조)\n"))
with open(f"{OUT}/SHA256SUMS", "w") as f:
    for n in ("phase6_review_documents.zip", "phase6_verification_bundle.zip"):
        f.write(f"{sha(f'{OUT}/{n}')}  {n}\n")
print("files", len(G), "excluded(hash only)", len(EXCL))
