"""Build Phase 5 review ZIPs (read-only on project files). Run from project root."""
import glob, hashlib, json, os, sys, time, zipfile, datetime

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def mt(p):
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")


# ---------- 1. documents zip ----------
DOCS = [("results/notes/phase5_mathematical_foundations.md", "Gate A 수학 기초: 정의, 명제 1–6, 대칭, Takens 적용표, 미해결 목록", "H1,H2,H3 (이론)"),
        ("results/notes/phase5_hypothesis_protocol.md", "사전 등록 프로토콜(10:50 고정, §9 변경 기록, §10 Gate F)", "H1,H2,H3,Gate F"),
        ("results/notes/phase5_controlled_experiment_results.md", "Gate E/F 결과 보고(사전 등록 + 사후 탐색 구분)", "H1,H2,H3,Gate F"),
        ("results/notes/phase5_observability_information_audit.md", "Gate C 관측 수준 감사(국소 Jacobian, 대칭, 전역 모호성, 물리 기준)", "H1 (관측 정보), Q1"),
        ("research_handover_latest.md", "인수인계 진입점 (패키지 생성 직전 버전)", "전체")]
with zipfile.ZipFile(f"{OUT}/phase5_review_documents.zip", "w", zipfile.ZIP_DEFLATED) as z:
    idx = [f"# FILE_INDEX — phase5_review_documents.zip\n\n생성: {now}. 원본 파일은 수정하지 않고 복사만 했다. 프로젝트 루트: `/home/minju/faicpaston/Imputing_Missing_Data`\n",
           "| ZIP 내 경로 | 실제 경로 | 역할 | 최종 수정 | 관련 가설 | sha256 |", "|---|---|---|---|---|---|"]
    for p, role, h in DOCS:
        z.write(p, p); idx.append(f"| `{p}` | `{os.path.abspath(p)}` | {role} | {mt(p)} | {h} | `{sha(p)[:16]}` |")
    z.writestr("FILE_INDEX.md", "\n".join(idx) + "\n")

# ---------- 2. verification bundle ----------
E = "results/extra/phase5"
G = []  # (path, role, hypothesis)
def add(pattern, role, hyp):
    for p in sorted(glob.glob(pattern)):
        if os.path.isfile(p): G.append((p, role, hyp))

# raw evaluation CSVs (source of all H1/H2/H3 numbers)
add(f"{E}/eval/*_probe.csv", "원본 평가: 잠재/원시 특징별 MLP·ridge probe의 test 궤적×층별 SSE/N (y_j, s_j)", "H1,H2,B−Sham(사후)")
add(f"{E}/eval/*_h3.csv", "원본 평가: 공유 readout의 마스크 5개별 오차와 분산", "H3")
add(f"{E}/eval/*_dyn.csv", "원본 평가: y_j(t+5) 동역학 probe", "2차 결과")
add(f"{E}/eval/*_imp.csv", "원본 평가: 자기 view 결측 위치 대치 SSE/N", "2차 결과(H2 보조)")
add(f"{E}/eval/scheduler.log", "평가 스케줄러 실행 로그(36/36, rc)", "실행 무결성")
# analysis outputs
add(f"{E}/analysis/contrasts_probe_mlp.csv", "사전 등록 대비표(주 대비 H2 포함): 평균, t-CI 95/90, Wilcoxon, crossed bootstrap, 판정", "H1,H2")
add(f"{E}/analysis/levels_probe_mlp.csv", "Arm/특징별 수준(복제본 평균·SD)", "H1,H2")
add(f"{E}/analysis/levels_ridge.csv", "ridge readout 수준(y, s)", "H2,H3")
add(f"{E}/analysis/h3_*.csv", "H3 수준과 대비", "H3")
add(f"{E}/analysis/dyn_levels.csv", "동역학 probe 수준", "2차 결과")
add(f"{E}/analysis/imputation_per_rep.csv", "대치 RMSE 복제본별", "2차 결과")
add(f"{E}/analysis/contrasts_imputation.csv", "대치 대비", "2차 결과")
add(f"{E}/analysis/seed_variability.csv", "학습 seed만의 변동(복제본 0, seed 0/1/2)", "H2 (변동 추정)")
add(f"{E}/analysis/training_rows.csv", "학습 304개 요약(파라미터 수, best_val, epoch, cap, nonfinite, code_md5, cfg)", "실행 무결성")
add(f"{E}/analysis/posthoc_exploratory.csv", "사후 탐색 결과: B−Sham, B−A, Sham−A 및 κ DiD (사전 등록 아님)", "B−Sham(사후)")
# H1 capacity
add(f"{E}/h1/*", "H1 용량 연구: 8궤적 합동, 폭 64/256/1024, 특징 own/joint/owncur/cur, test 궤적×층별 SSE/N 및 로그", "H1")
# Gate C
add(f"{E}/gateC/*", "Gate C 관측 감사 산출물(국소 Jacobian, 대칭, 자료 점검, 물리 기준, 전역 모호성)", "H1/Q1 (관측 수준)")
# Gate F
add(f"{E}/external/*.npz", "Gate F 원본 평가: 모델별 주 블록×변수×결측길이 층 SSE/N (MV/MVshift/SV_k/기준선; info=학습 메타)", "Gate F")
add(f"{E}/external/analysis/*.csv", "Gate F 분석 결과(대비, 수준, 요약)", "Gate F")
add(f"{E}/external/lanes/*", "Gate F 작업 목록과 lane 로그(rc)", "Gate F 실행 무결성")
# training manifests / logs
add(f"{E}/runs/rows/*.csv", "학습 1개당 결과 행(설정, best_val, epoch, md5, GPU)", "실행 Manifest")
add(f"{E}/runs/logs/*", "학습 1개당 epoch 로그", "실행 Manifest")
add(f"{E}/runs/lanes/*", "Gate E 작업 목록(main_jobs.txt, part 파일)과 lane 로그(rc)", "실행 Manifest")
# code
for p, role, h in [
    ("experiments/phase5/analyze.py", "사전 등록 분석 스크립트(H1/H2/H3, 판정 규칙, δ=0.02)", "H1,H2,H3"),
    ("experiments/phase5/analyze_posthoc.py", "사후 탐색 분석(B−Sham, κ DiD) — 사전 등록 아님", "B−Sham(사후)"),
    ("experiments/phase5/analyze_external.py", "Gate F 사전 등록 분석(seed별 주 블록 bootstrap)", "Gate F"),
    ("experiments/phase5/core.py", "Arm 정의(arm_spec), 자료 조립, 정규화, 마스크, ArmModel, seed_of", "H2 설계"),
    ("experiments/phase5/train.py", "Arm 학습(CFG, FDR 손실, 참값 없는 조기 종료)", "H2 설계"),
    ("experiments/phase5/evaluate.py", "probe/H3/동역학/대치 평가", "H1,H2,H3"),
    ("experiments/phase5/h1_capacity.py", "H1 용량 연구", "H1"),
    ("experiments/phase5/gen_data.py", "독립 궤적 생성(SEED_BASE 20261028)", "자료"),
    ("experiments/phase5/obs_audit.py", "Gate C 관측 감사", "Q1/H1"),
    ("experiments/phase5/external.py", "Gate F 전처리·학습·shift·채점", "Gate F"),
    ("experiments/phase5/eval_scheduler.py", "평가 스케줄러", "실행"),
    ("experiments/phase5/run_lane.sh", "Gate E lane 실행기", "실행"),
    ("experiments/phase5/run_lane_cmd.sh", "Gate F lane 실행기", "실행"),
    ("experiments/phase5/tests/test_phase5.py", "무결성 테스트 8개(마스크 동일, hidden 미입력, train 정규화, past-only, Bcur, Sham, 손실 목표)", "무결성"),
    ("experiments/phase5/tests/test_external.py", "Gate F 무결성 테스트 3개(shift, 제외, 층)", "무결성"),
    # shared dependencies imported by phase5 code
    ("experiments/fdr/loss.py", "의존 모듈: fdr_masked_loss, sample_hide", "H2 설계"),
    ("src/models.py", "의존 모듈: group_mlp, mlp", "H2 설계"),
    ("src/lorenz.py", "의존 모듈: Lorenz 적분", "자료"),
    ("src/delay_embedding.py", "의존 모듈: 지연 매장", "자료")]:
    if os.path.isfile(p):
        G.append((p, role, h))

# manifest of excluded large files (hash only)
excl = []
for p in sorted(glob.glob(f"{E}/data/*.npz")) + sorted(glob.glob("Data/external/*/processed/*.npz")):
    excl.append(dict(path=p, bytes=os.path.getsize(p), sha256=sha(p), mtime=mt(p), reason="대용량 시계열(재생성 가능: gen_data.py / external.py prep)"))
ck = sorted(glob.glob(f"{E}/runs/ckpt/*"))
manifest = dict(created=now, project_root=os.path.abspath("."), python="~/.conda/envs/reef_gpu/bin/python, PYTHONPATH=.pydeps:.",
                note="원본 파일은 수정하지 않고 복사만 했다. 모든 수치 파일은 생성 이후 변경되지 않았다(mtime 참고).",
                rerun=["PYTHONPATH=.pydeps:. python experiments/phase5/analyze.py",
                       "python experiments/phase5/analyze_posthoc.py", "python experiments/phase5/analyze_external.py",
                       "python experiments/phase5/tests/test_phase5.py; python experiments/phase5/tests/test_external.py  (스크립트 실행, 궤적/외부 자료 필요)"],
                included=[dict(path=p, bytes=os.path.getsize(p), sha256=sha(p), mtime=mt(p), role=r, hypothesis=h) for p, r, h in G],
                excluded_large=excl,
                excluded_checkpoints=dict(dir=f"{E}/runs/ckpt", n_files=len(ck), total_bytes=sum(os.path.getsize(c) for c in ck)))

# FILE_INDEX.md: group by role (one row per group for bulk files, one row per file otherwise)
from collections import OrderedDict
grp = OrderedDict()
for p, r, h in G:
    grp.setdefault((r, h), []).append(p)
L = [f"# FILE_INDEX — phase5_verification_bundle.zip\n",
     f"생성: {now}. 프로젝트 루트 `/home/minju/faicpaston/Imputing_Missing_Data`. ZIP 내 경로 = 프로젝트 루트 기준 실제 상대 경로.",
     "원본은 수정하지 않고 복사만 했다. 파일별 sha256·크기·mtime은 `MANIFEST.json`에 있다. 새 연구 결론은 포함하지 않는다.\n",
     "## 읽는 순서 (검증 경로)",
     "1. 주 대비 H2 (κ=1, p=0.7, B−A): `experiments/phase5/analyze.py` → `results/extra/phase5/analysis/contrasts_probe_mlp.csv` (원본: `eval/*_probe.csv`, feature A1/A2/B1/B2, probe mlp0/mlp1, target y1/y2)",
     "2. 복제본별 값: `analyze_posthoc.py` 의 R 테이블(복제본×Arm RMSE)을 같은 원본에서 재계산; 학습 seed 변동은 `seed_variability.csv`",
     "3. B−Sham(사후 탐색, 사전 등록 아님): `analyze_posthoc.py` → `posthoc_exploratory.csv`",
     "4. H1: `contrasts_probe_mlp.csv`(raw_* 특징) + `h1/*.csv`; 관측 수준 근거 `gateC/*`",
     "5. H3: `eval/*_h3.csv` → `analysis/h3_*.csv`",
     "6. Gate F: `external/*.npz` → `analyze_external.py` → `external/analysis/*.csv`",
     "7. 실행 무결성: `analysis/training_rows.csv`, `runs/rows`, `runs/lanes/*.log`, `external/lanes/*.log`, `eval/scheduler.log`, 테스트 코드\n",
     "## 파일 이름 규칙",
     "- `eval/k{κ}_p{p}_r{복제본}_s{학습seed}_{probe|h3|dyn|imp}.csv`: 주 복제본은 s = r, 추가 seed는 r0의 s1, s2.",
     "- `runs/*/k{κ}_p{p}_r{r}_{Arm}_s{seed}`: Arm ∈ A1,A2,B1,B2,Bcur1,Bcur2,C,Sham1,Sham2.",
     "- `external/{ds}_p{p}_s{seed}_{model}.npz`: SSE, N 배열 (주 블록, 변수, 결측길이 층 g∈{1,2–3,4–7,≥8}).\n",
     "## 파일 목록\n", "| 실제 경로 (패턴) | 파일 수 | 역할 | 생성/수정 시점 범위 | 관련 가설 |", "|---|---|---|---|---|"]
for (r, h), ps in grp.items():
    ts = sorted(mt(p) for p in ps); when = ts[0] if len(ts) == 1 else f"{ts[0]} ~ {ts[-1]}"
    name = ps[0] if len(ps) == 1 else (os.path.dirname(ps[0]) + "/" + ("*" + os.path.splitext(ps[0])[1] if len({os.path.splitext(x)[1] for x in ps}) == 1 else "*"))
    if 1 < len(ps) <= 3:
        name = "`, `".join(ps)
    if len(ps) > 1 and len(grp) and r.startswith("원본 평가") and "_" in ps[0]:
        name = os.path.dirname(ps[0]) + "/*_" + ps[0].rsplit("_", 1)[1]
    L.append(f"| `{name}` | {len(ps)} | {r} | {when} | {h} |")
L += ["\n## 제외한 파일 (해시만 MANIFEST.json에 기록)",
      f"- 학습 궤적 `{E}/data/*.npz` ({len([e for e in excl if e['path'].startswith(E)])}개) 및 외부 전처리 시계열 `Data/external/*/processed/*.npz`: 재생성 가능, sha256 기록.",
      f"- 체크포인트 `{E}/runs/ckpt/` ({len(ck)}개, {manifest['excluded_checkpoints']['total_bytes']/1e6:.1f} MB).",
      "- 외부 원자료 `Data/external/*/raw` (읽기 전용, 감사 문서 참고).",
      "\n## 재실행", *[f"- `{c}`" for c in manifest["rerun"]],
      "\n## 주의 (기존 문서에 이미 기록된 사항의 재기술, 새 결론 아님)",
      "- 사전 등록 분석의 판정 함수는 TOST를 OPPOSITE보다 먼저 검사한다(결과 문서 §1 메모).",
      "- `posthoc_exploratory.csv`와 `analyze_posthoc.py`는 주 결과를 본 뒤 작성되었다.",
      "- 파일럿 9개 학습은 code_md5 f859de28 (A2 출력 인덱스 수정 전), 나머지는 a293c037 (프로토콜 §9).",
      "\n## 패키지 생성 시 재현 점검 (2026-10-28 13:10–13:15, CPU)",
      "- 이 ZIP만 풀어 `analyze_posthoc.py` 실행 → `posthoc_exploratory.csv` 원본과 바이트 동일.",
      "- 이 ZIP만 풀어 `analyze_external.py` 실행 → `external/analysis/{contrasts,levels,summary}.csv` 원본과 바이트 동일.",
      "- `analyze.py`는 ZIP 내 `eval/*.csv`만 사용한다(이번 점검에서는 재실행하지 않음, 원 실행 약 4분 20초).",
      "- 무결성 테스트는 궤적 자료(`results/extra/phase5/data/*.npz`, 제외됨)가 필요하다. `python experiments/phase5/gen_data.py`로 재생성 후 `python experiments/phase5/tests/test_phase5.py`(8) 및 `test_external.py`(3, 외부 원자료 필요). 원 프로젝트에서 13:13에 다시 실행해 11/11 PASS 확인.",
      "- 실행 환경: `PYTHONPATH=.` (`.pydeps`는 포함하지 않았다; torch, numpy, pandas, scipy 필요)."]
with zipfile.ZipFile(f"{OUT}/phase5_verification_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p, _, _ in G:
        z.write(p, p)
    z.writestr("FILE_INDEX.md", "\n".join(L) + "\n")
    z.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
print(len(G), "files in bundle;", len(excl), "excluded large;", len(ck), "ckpt")
