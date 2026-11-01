#!/usr/bin/env python
"""Build the offline research dashboard index.html at the archive root.
Every number shown in a chart or results table is read here from a CSV that is INSIDE the archive; nothing is typed by hand except
qualitative text. Links are emitted only for archive files that exist (checked here); otherwise the path is shown as plain text.
usage: python build_dashboard.py <archive root>"""
import csv, json, math, os, sys
from collections import defaultdict

R = sys.argv[1]; os.chdir(R); MISSING = []


def L(p, label=None):
    """link object if the file exists inside the archive, else plain text marker."""
    ok = os.path.exists(p)
    if not ok:
        MISSING.append(p)
    return dict(path=p, label=label or os.path.basename(p.rstrip("/")), ok=ok)


def rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def ci(vals):
    n = len(vals); m = sum(vals) / n; sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    t = {24: 2.0687, 8: 2.3646, 6: 2.5706}.get(n, 2.0)                      # two-sided 95% t quantiles (df = n - 1)
    return dict(mean=m, lo=m - t * sd / math.sqrt(n), hi=m + t * sd / math.sqrt(n), n=n)


f = lambda x: float(x)
# ---------------- Phase 9 (main) imputation and retrieval, per cell, replicate-level -> mean and 95% CI
P9I = "04_Core_Experiments/Phase9/analysis_main/p9_imputation_rep.csv"; P9R = "04_Core_Experiments/Phase9/analysis_main/p9_retrieval_rep.csv"
def p9chart(path, cols, ent):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows(path):
        for c, e in zip(cols, ent):
            d[(f(r["kappa"]), f(r["p"]))][e].append(f(r[c]))
    out = {}
    for p in (0.7, 0.3):
        out[str(p)] = [dict(label=f"κ = {int(k)}", bars=[dict(entity=e, **ci(d[(k, p)][e])) for e in ent]) for k in (0.0, 1.0)]
    return out
CH_P9_IMP = p9chart(P9I, ["AE:P9", "AE:Bfull", "AE:Fact8"], ["P9-MVAE", "B-full", "Fact8"])
CH_P9_RET = p9chart(P9R, ["P9", "Bfull", "Fact8"], ["P9-MVAE", "B-full", "Fact8"])
# ---------------- Phase 8 Family III imputation (independent recomputation, replicate level)
P8I = "05_Imputation_Results/Phase8_FamilyIII/indep_familyIII_rep_level.csv"
d8 = defaultdict(lambda: defaultdict(list))
for r in rows(P8I):
    if r["method"] in ("AE:Bfull", "AE:Fact8"):
        d8[(f(r["kappa"]), f(r["p"]))]["B-full" if r["method"] == "AE:Bfull" else "Fact8"].append(f(r["rmse_pooled"]))
CH_P8_IMP = {str(p): [dict(label=f"κ = {int(k)}", bars=[dict(entity=e, **ci(d8[(k, p)][e])) for e in ("B-full", "Fact8")]) for k in (0.0, 1.0)] for p in (0.7, 0.3)}
# ---------------- Phase 7: RMSE_s6 B-full (mean_a) vs B-self (mean_b) per cell + contrast CI
P7 = "04_Core_Experiments/Phase7/analysis/p7_fixed_sequence.csv"; r7 = {(f(r["kappa"]), f(r["p"])): r for r in rows(P7)}
CH_P7 = {str(p): [dict(label=f"κ = {int(k)}", bars=[dict(entity="B-self", mean=f(r7[(k, p)]["mean_b"]), n=24), dict(entity="B-full", mean=f(r7[(k, p)]["mean_a"]), n=24)],
                       note=f"Δ = B-full − B-self = {f(r7[(k, p)]['mean']):.4f} [{f(r7[(k, p)]['ci95_lo']):.4f}, {f(r7[(k, p)]['ci95_hi']):.4f}]") for k in (0.0, 1.0)] for p in (0.7, 0.3)}
# ---------------- confirmatory / key tests table (all read from CSV)
T = []
for r in rows("04_Core_Experiments/Phase5/analysis/contrasts_probe_mlp.csv"):
    if r["contrast"] == "H2 primary: B - A" and "'kappa': np.float64(1.0), 'p': np.float64(0.7), 'stratum': 'all'" in r["setting"]:
        T.append(dict(phase="5", test="H2 B − A (κ=1, p=0.7, y_j RMSE)", mean=f(r["mean"]), lo=f(r["ci95_lo"]), hi=f(r["ci95_hi"]), unit="y_j RMSE, n=8", status="fail",
                      decision="지지 안 됨(CI가 0 포함, 방향 반대)", src=L("04_Core_Experiments/Phase5/analysis/contrasts_probe_mlp.csv")))
g6 = {r["step"]: r for r in rows("04_Core_Experiments/Phase6/analysis/g2_fixed_sequence.csv")}
s = g6["S1 B-Sham k1 p0.7"]; T.append(dict(phase="6", test="S1 B − Sham (κ=1, p=0.7)", mean=f(s["mean"]), lo=f(s["ci95_lo"]), hi=f(s["ci95_hi"]), unit="y_j RMSE, n=24", status="hold",
                                            decision="검출, 그러나 κ=0 음성 대조 실패로 보류", src=L("04_Core_Experiments/Phase6/analysis/g2_fixed_sequence.csv")))
for r in rows("04_Core_Experiments/Phase6/analysis/g2_contrasts.csv"):
    if r["kappa"] == "0.0" and r["p"] == "0.7" and r["stratum"] == "all" and r["contrast"] == "B - Sham":
        T.append(dict(phase="6", test="음성 대조 B − Sham (κ=0, p=0.7)", mean=f(r["mean"]), lo=f(r["ci95_lo"]), hi=f(r["ci95_hi"]), unit="y_j RMSE, n=24", status="fail",
                      decision="실패(0이어야 함), 원인 미해결", src=L("04_Core_Experiments/Phase6/analysis/g2_contrasts.csv")))
for r in rows("04_Core_Experiments/Phase6/s1r_results.csv"):
    if r["kappa"] == "1.0":
        T.append(dict(phase="6", test="S1R B − Sham (κ=1, 재현 설계)", mean=f(r["mean"]), lo=f(r["ci95_lo"]), hi=f(r["ci95_hi"]), unit="y_j RMSE, n=24", status="desc",
                      decision="검출(사후 설계 재현; 확증 아님), δ=0.02 미만", src=L("04_Core_Experiments/Phase6/s1r_results.csv")))
for k, p in ((1.0, 0.7), (0.0, 0.7)):
    r = r7[(k, p)]; T.append(dict(phase="7", test=f"Bfull − Bself RMSE_s6 (κ={int(k)}, p={p})", mean=f(r["mean"]), lo=f(r["ci95_lo"]), hi=f(r["ci95_hi"]), unit="RMSE_s6, n=24",
                                  status="pass", decision="확증(κ=0에서도 같음 → 결합 효과 아님)", src=L(P7)))
A8 = "04_Core_Experiments/Phase8/analysis_complete"
for r in rows(f"{A8}/p8_familyI_sequence.csv")[:2]:
    T.append(dict(phase="8", test=f"{r['step']} {r['contrast']} (κ=1, p=0.7, H층)", mean=f(r["mean"]), lo=f(r["lcb"]), hi=f(r["ucb"]), unit="MSE, n=24 (97.5% 단측)",
                  status="pass" if r["confirmed_gt_delta"] == "True" else "fail", decision="확증(LCB > δ_I = 0.02)" if r["confirmed_gt_delta"] == "True" else "실패(δ_I 근방)",
                  src=L(f"{A8}/p8_familyI_sequence.csv")))
for r in rows(f"{A8}/p8_familyII_sequence.csv"):
    st = "pass" if r["confirmed"] == "True" else ("untested" if r["tested"] == "False" else "fail")
    dec = {"IIa-1": "확증(δ_y = 0.02 미달)", "IIa-2": "확증(P8.4b 인증)", "IIb-1": "미검출(경계)", "IIb-2": "미검정(서술: B-full이 더 나쁨)"}[r["step"]]
    T.append(dict(phase="8", test=f"{r['step']} {r['contrast']} (κ=1, p=0.7)", mean=f(r["mean"]), lo=f(r["lcb"]), hi=f(r["ucb"]), unit="RMSE 계열, n=24 (98.75% 단측)", status=st, decision=dec,
                  src=L(f"{A8}/p8_familyII_sequence.csv")))
for r in rows("04_Core_Experiments/Phase9/analysis_main/p9_confirmatory_sequence.csv"):
    st = "pass" if r["decision"] == "True" else ("untested" if r["tested"] == "False" else "fail")
    nm = {"P-1": "대치 P9 − B-full", "P-2": "ρ_ret P9 − B-full", "P-3": "대치 P9 − Fact8 (비열등 η=0.01)", "P-4": "ρ_ret P9 − Fact8"}[r["step"]]
    T.append(dict(phase="9", test=f"{r['step']} {nm} (κ=1, p=0.7)", mean=f(r["mean"]), lo=f(r["lcb"]), hi=f(r["ucb"]), unit="n=24 (97.5% 단측)", status=st,
                  decision={"P-1": "실패(반대 방향)"}.get(r["step"], "미검정(순서 규칙)"), src=L("04_Core_Experiments/Phase9/analysis_main/p9_confirmatory_sequence.csv")))
J = defaultdict(lambda: defaultdict(list))
for r in rows("05_Imputation_Results/Phase9_Jena_real_data/jena_rep_rmse.csv"):
    for m in ("AE:P9", "AE:Bfull", "AE:SelfV", "LOCF", "zero"):
        J[r["p"]][m].append(f(r[m]))
JENA = [dict(p=p, **{m: sum(v) / len(v) for m, v in J[p].items()}) for p in sorted(J)]

# ---------------- qualitative content (paths are checked)
PH = [
 dict(id="1", title="Phase 1 — 초기 다중 view 지연 AE와 FPRM 전제 점검", date="10-24 → 10-27",
      q="완전 관측 변수 없이(모든 변수 MCAR) 다중 view 지연 AE가 결측을 대치할 수 있는가? FPRM의 '완전 관측 참조 1개' 전제는 성립하는가?",
      math="Takens/SYC(문헌), z만 쓰는 Lorenz 지연 사상의 비단사성(Γ 대칭), 결측 지연 벡터는 임베딩을 물려받지 않음",
      model="A(현재 시점 MLP), B(변수별 지연 AE), C(view별 인코더 + 평균 fusion, 공유 잠재 16), E(view dropout), F(학습 fusion)",
      exp="Lorenz(dt 0.01/0.05/0.1), 결합 Lorenz, 실제 자료 5종; MCAR p 0.1–0.7; seed 3개(단일 궤적 → 독립 복제 아님)",
      res="조건 B(참조 없음, dt=0.05)에서 C 0.0091/0.0179/0.0387/0.1424 대 linear(과거+미래 사용) 0.1472/0.2130/0.3517/0.5713. z 참조 FPRM-T의 nRMSE 1.006–1.069(대칭 실패)",
      neg="C의 단일 view 경로가 약함. view dropout은 다변량 정확도를 해침. ENSO와 Exchange에서는 linear가 최선. FPRM 논문 수치 미재현",
      interp="참조 없는 대치는 가능하지만, 공유 잠재의 일관성은 달성되지 않았다", files=[L("04_Core_Experiments/Phase1_2_early/aim_autonomous_20261027.md"), L("04_Core_Experiments/Phase1_2_early/math_review.md"), L("04_Core_Experiments/Phase1_2_early/observations.md"), L("04_Core_Experiments/Phase1_2_early/results/real_data_results.csv")]),
 dict(id="2", title="Phase 2 / 2.5 — 부분 관측 식별성의 수학과 감사", date="10-27",
      q="부분 관측 식별성을 정식화하고, 일관된 공유 잠재를 위한 구조를 설계할 수 있는가?", math="P2 Γ 비식별, P4 결합 식별, **P7 점 일관성 → 정보 붕괴**, P8 평균 fusion 딜레마, P10 tower(Phase 2.5 감사로 일부 철회)",
      model="설계만 함(FDR-AE, XV-AE, PoE-FDR)", exp="CPU 진단(누출 섭동 시험 출력 차이 0.0)", res="결정: 결정론적 FDR을 먼저 구현",
      neg="MCAR ⇒ PoE 정확 주장 철회. 'ELBO 최적 ⇒ 참 사후분포' 틀림. PoE-FDR 권고 철회", interp="무조건 일관성 강제는 정보를 붕괴시킨다는 원리(이후 Phase 9 A2.2로 정리)",
      files=[L("04_Core_Experiments/Phase1_2_early/aim_mathematical_foundation.md"), L("04_Core_Experiments/Phase2_5/phase25_mathematical_audit.md"), L("04_Core_Experiments/Phase2_5/phase25_architecture_decision.md")]),
 dict(id="3", title="Phase 3 — Full Delay Reconstruction(FDR)과 평가 오류 정정", date="10-27",
      q="전체 지연 재구성(FDR)이 숨은 상태 정보와 일관성을 개선하는가?", math="Γ와 y = x + ẋ/σ(관측 가능성), 디코더 섬유 진단(P11′)",
      model="C 대 FDR(1/c(u) 중복도 가중), lag 가중 α", exp="Lorenz(x, z 관측, y 숨김); 단일 궤적 seed 3–5개", res="3D-2: FDR(α=1/8)은 대치 5/5 악화, y-probe 5/5 개선 → trade-off. 숨은 상태 이득은 지연 목표에서 온다",
      neg="**3C의 'FDR이 같은 상태 일관성 개선'은 무효**(관측 좌표가 섞인 지표; 3D-1이 정정). lag 가중으로 trade-off를 해소하지 못함", interp="평가 지표는 결측 좌표만 써야 한다. 대치 정확도와 상태 정보는 다르다",
      files=[L("04_Core_Experiments/Phase3/phase3d1_evaluation_integrity.md"), L("04_Core_Experiments/Phase3/phase3d2_fdr_lag_weight_results.md"), L("04_Core_Experiments/Phase3/phase3c_experiment_results.md")]),
 dict(id="4", title="Phase 4 — 결합 Lorenz 파일럿", date="10-27", q="불완전한 다중 view 지연(x1, z1, x2, z2)을 결합하면 식별과 복원이 개선되는가?",
      math="κ=0에서 하위계 view rank 3, κ=1에서 rank 6이지만 증폭이 큼. h±: 국소 rank 6이면서 전역 비단사인 반례", model="decoder {C, FDR} × fusion {Scalar, Learned}",
      exp="κ{0,1}, p{0.3,0.7}, 단일 궤적 seed 3개", res="p=0.7에서 FDR − C −0.022 ~ −0.035, p=0.3에서는 C가 나음(부호 반전)", neg="view 제거 비교가 분포 밖(OOD)이었음 → Phase 5 설계로 대체. p=0.7에서 잠재 이점 없음",
      interp="통제된 Arm 비교와 독립 복제가 필요하다", files=[L("04_Core_Experiments/Phase4/phase4_coupled_pilot_results.md"), L("04_Core_Experiments/Phase4/phase4_coupled_observability.md")]),
 dict(id="5", title="Phase 5 — 통제 다중 view 실험(Own vs Multi-view, Sham)", date="10-28", q="상대 하위계 view 입력(Arm B)이 자기 숨은 y를 Arm A(자기 view만)보다 잘 복원하는가?",
      math="Lemma S(대칭), Prop. 3(κ=0 교차 Bayes 이득 0), Prop. 5(κ=1 순간 항등식), DPI", model="FDR-Learned ArmModel: A, B, Bcur, Sham, C", exp="κ{0,1} × p{0.3,0.7}, 복제본 8개(test 궤적 공유), 실제 자료 Beijing, Jena",
      res="H2 B − A(κ=1, p=0.7) +0.0137 [−0.0015, +0.0288] → 지지 안 됨. Beijing p=0.7 MV − MV-shift −0.137", neg="입력 비용이 이득을 압도함. 정오표 E6(donor 재사용으로 Sham CI 무효), E7(MV-shift가 과거 전용이 아님)",
      interp="정보 존재와 활용을 분리해야 한다", files=[L("04_Core_Experiments/Phase5/phase5_controlled_experiment_results.md"), L("04_Core_Experiments/Phase5/phase5_mathematical_foundations.md"), L("04_Core_Experiments/Phase5/analysis/contrasts_probe_mlp.csv")]),
 dict(id="6", title="Phase 6 — 정보 하한과 Real vs Sham", date="10-28", q="정보 존재(H1), 사용(H2), 좌표(H3)를 분리할 수 있는가?",
      math="L0 대칭: E[y_j | U_j] = 0, Δ* ≥ p⁸Δ*_L0, 교차 적합 편향, Prop. E3.1(κ=0 donor 교환가능성)", model="지도 G1 추정기, AE A/B/Sham, 6차원 readout", exp="새 궤적, 복제본 24개, κ{0,1} × p{0.3,0.7}",
      res="G1 Δ*_L0 LCB 0.0322(→ Empirical로 강등). S1 B − Sham −0.0070, S1R −0.0104", neg="**κ=0 음성 대조 실패 +0.0062(미해결)**. G3 공유 좌표 기준 미달. S1R/S3 '확증' 주장 하향",
      interp="상대 view 정보 존재를 처음으로 엄밀하게 보였다. 활용은 작다", files=[L("04_Core_Experiments/Phase6/phase6_final_report.md"), L("04_Core_Experiments/Phase6/phase6_g1_results.md"), L("04_Core_Experiments/Phase6/phase6_mathematical_errata.md"), L("04_Core_Experiments/Phase6/analysis/g2_fixed_sequence.csv")]),
 dict(id="7", title="Phase 7 — Full-target supervision(B-self 대 B-full)", date="10-28/29", q="같은 구조에서 손실 열만 바꾸면(B-full) 잠재가 6차원 결합 상태를 더 잘 보존하는가?",
      math="A1(Δ* ≥ p⁸G_L0 증명; Δ* > 0은 Empirical), Prop. P7.1, DPI", model="Bself1, Bself2, Bfull(같은 입력·디코더·초기화, 손실 열만 다름)", exp="새 id 40000+, 복제본 24 × 4셀",
      res="RMSE_s6 B-full − B-self = −0.3702(κ=1, p=0.7), 24/24. 네 셀 모두 확증", neg="**κ=0에서도 같은 이득 → 결합 활용이 아니라 감독 범위 효과**. p=0.7 G3 기준 실패. L0층 y는 영 예측보다 나쁨(1.019)",
      interp="잠재가 무엇을 담는지는 학습 목표가 결정한다", files=[L("04_Core_Experiments/Phase7/phase7_final_report.md"), L("04_Core_Experiments/Phase7/phase7_mathematical_corrections.md"), L(P7)]),
 dict(id="8", title="Phase 8 — 정보 한계 대 모델 한계, CSIA, B-full 대 Fact8", date="10-29 → 10-31", q="(I) 어려운 층의 실패는 정보 한계인가 모델 한계인가? (II) 결합 인코더가 병치보다 결합 특이 정보를 더 잘 실현하는가?",
      math="B1/B3 Bayes 부등식, N1 대칭 하한(X0 LB 0.6679), P8.1–P8.4, 반례 C1–C5", model="B-full, BfullSham, Fact8, Fact16, oracle(O_raw, O_lat 등), readout R1", exp="복제본 24 × 4셀, 작업 1,276개, FWER 통제",
      res="I-1 모델 한계 ≥ 0.041(확증). IIa-1 −0.0086, IIa-2 0.033(확증)", neg="**IIb-1 미검출**: B-full이 6차원 decodability(+0.018)와 p=0.7 대치에서 Fact8보다 나쁨(서술). I-2 실패. GSR 미충족",
      interp="정보는 있고 일부 활용되지만, 결합 공유 잠재의 우위는 없다", files=[L("04_Core_Experiments/Phase8/phase8_postrun_independent_audit_20261031.md"), L("04_Core_Experiments/Phase8/phase8_mathematical_conclusions_20261031.md"), L("04_Core_Experiments/Phase8/phase8_mathematical_protocol_v2.md")]),
 dict(id="9", title="Phase 9 — 수학적으로 정당화된 공유 상태 학습(P9-MVAE)", date="10-31", q="tower 자기 증류(teacher가 전체 관측 O를 보고 원래 결측 위치에만 목표를 줌)가 공유 상태 학습과 대치를 개선하는가?",
      math="A2.2(무조건 정렬 불가), A3.1/A3.2(조건부 임베딩), A4.1(참조 없는 학습의 Fisher 일관성), A4.3(유일 고정점, 붕괴 조건)", model="P9-MVAE = B-full + L_tow(λ=1, no-grad teacher)",
      exp="사전등록·고정, pilot(복제본 6) 통과, main(새 id 200000+, 복제본 24 × 4셀 × 모델 6개), Jena 실제 자료", res="**P-1 실패**: 대치 P9 − B-full +0.0021(반대 방향). P-2–P-4 미검정. ρ_ret −0.004(미미)",
      neg="p=0.3에서 대치 악화(+0.0078). 학습 발산 3/24. Fact8 > B-full 재현. Jena에서 P9 8/8 악화", interp="추정 대상을 보존하는 메커니즘은 유한 용량 이득에만 의존하는데, 그 이득이 나타나지 않았다",
      files=[L("04_Core_Experiments/Phase9/phase9_mathematical_foundations.md"), L("04_Core_Experiments/Phase9/phase9_main_results.md"), L("04_Core_Experiments/Phase9/phase9_mathematical_conclusions.md"), L("04_Core_Experiments/Phase9/phase9_preregistered_protocol.md")]),
]
MD = "02_Mathematical_Foundations/source_documents/"
MATH = dict(
 proven=[("Lorenz 대칭군: γ₁₂는 모든 κ에서, γ_j는 κ=0에서만 흐름과 가환", MD + "phase8_mathematical_protocol_v2.md"), ("z만 쓰는 지연 사상은 적어도 2:1(비단사)", MD + "math_review.md"),
         ("N1-X0 대칭 하한 R*_X0 ≥ (1/6)Σ E s²/σ²", MD + "phase8_mathematical_protocol_v2.md"), ("B1: R(h) − R* ≥ R(h) − R(f); B3: 잠재와 raw 차이 ≠ Bayes 손실", MD + "phase8_mathematical_protocol_v2.md"),
         ("P8.1 κ=0 인수분해, P8.4 인증 정리", MD + "phase8_mathematical_protocol_v2.md"), ("P7: 점 일관성 → 정보 붕괴", MD + "aim_mathematical_foundation.md"),
         ("A2.2: 무조건 cross-mask 정렬과 상태 구별성은 양립 불가", MD + "phase9_mathematical_foundations.md"), ("A3.1: 정확한 등식이면 T는 C¹ 임베딩(조건부); A3.2 근사 단사성", MD + "phase9_mathematical_foundations.md"),
         ("A4.1/A4.2: holdout 손실의 Fisher 일관성(참조 없는 학습)", MD + "phase9_mathematical_foundations.md"), ("A4.3: tower 목적의 유일 고정점, 참값 항목 없으면 붕괴", MD + "phase9_mathematical_foundations.md")],
 assumptions=[("MCAR: 마스크가 상태와 독립(코드로 확인)", MD + "phase9_mathematical_foundations.md"), ("양성 조건 P(H_u = 1 | B = b) = (1−p)h/(p+(1−p)h) > 0", MD + "phase9_mathematical_foundations.md"),
              ("평가 법칙 P(유한 시간) ≠ 불변측도 ν", MD + "phase8_mathematical_protocol_v2.md"), ("(미검증) 고정 좌표 관측 Φ의 임베딩성과 Lipschitz 역", MD + "phase9_mathematical_foundations.md")],
 counter=[("C1: L_j ≥ 2이어도 식별 불가능할 수 있음(P에서)", MD + "phase8_mathematical_protocol_v2.md"), ("C2: 국소 full rank ≠ 전역 단사(z만 2:1, X0 γ 궤도)", MD + "phase8_mathematical_protocol_v2.md"),
          ("C3: 좋은 대치 ≠ 숨은 상태 복원", MD + "phase8_mathematical_protocol_v2.md"), ("C4: 마스크 안정성 ≠ 식별", MD + "phase8_mathematical_protocol_v2.md"), ("C5: 주변 충분성 ≠ 결합 충분성", MD + "phase8_mathematical_protocol_v2.md"),
          ("h±: 국소 rank 6이면서 전역 비단사(결합 Lorenz)", MD + "phase4_coupled_observability.md")],
 cert=[("Phase 6 G1: Δ*_L0 LCB 0.0322(Empirical로 강등)", MD + "phase7_mathematical_corrections.md"), ("Phase 8 I-1: H층 모델 한계 ≥ 0.041 MSE(풀 조건부)", "04_Core_Experiments/Phase8/phase8_postrun_independent_audit_20261031.md"),
       ("Phase 8 IIa-2: φ_Bfull(W)는 L_j0에서 W_j의 함수가 아님", "04_Core_Experiments/Phase8/phase8_postrun_independent_audit_20261031.md"), ("Phase 8 X0: B-full 위험의 ≥ 83%가 정보 한계(LB 0.6679, 2/3 아님)", "04_Core_Experiments/Phase8/phase8_postrun_independent_audit_20261031.md")],
 unresolved=[("고정 좌표 관측의 임베딩성(H2)", "02_Mathematical_Foundations/unresolved_questions.md"), ("H∖X0의 정량적 하한", "02_Mathematical_Foundations/unresolved_questions.md"),
             ("출력 일관성 → 잠재 상태 일관성 조건", "02_Mathematical_Foundations/unresolved_questions.md"), ("유한 모델에서 결합 < 병치의 원인", "02_Mathematical_Foundations/unresolved_questions.md"),
             ("ν에서의 식별성", "02_Mathematical_Foundations/unresolved_questions.md"), ("반-경사 자기 증류의 안정성", "02_Mathematical_Foundations/unresolved_questions.md"), ("Phase 6 κ=0 음성 대조 실패 원인", "02_Mathematical_Foundations/unresolved_questions.md")],
 withdrawn=[("MCAR ⇒ 가우스 PoE 정확", MD + "phase25_mathematical_audit.md"), ("ELBO 최적 ⇒ 참 사후분포", "04_Core_Experiments/Phase2_5/phase25_objective_redesign.md"), ("Phase 3C 'FDR 일관성 개선'", "04_Core_Experiments/Phase3/phase3d1_evaluation_integrity.md")])
MATH = {k: [dict(text=t, link=L(p)) for t, p in v] for k, v in MATH.items()}
AIM = [("Dynamics-informed imputation", "partial", "지연 매장 기반 대치가 LOCF와 zero보다 크게 나음. 다만 동역학 정보 사용의 우위는 병치 대비 미확립"),
       ("No fully observed reference variable", "established", "A4.1(수학) + 모든 모델이 참조 없이 학습(코드)"),
       ("Multi-view delay embedding", "partial", "상대 view 정보 존재(인증)와 활용(작음, 확증). Jena에서 다중 view > 변수별 AE(서술)"),
       ("Consistent shared latent state", "unresolved", "G3·GSR·ρ_ret 기준 미달. P9 개입 효과 미미. 비식별 마스크의 원리적 한계"),
       ("Mathematical correspondence & identifiability", "unresolved", "조건부 정리(A3.1)만 있음. 학습 모델이 가정을 만족한다는 증거 없음"),
       ("Simulated chaotic systems & real-world data", "partial", "결합 Lorenz는 사전등록 확증. 실제 자료는 서술(FPRM 자료 5종, Beijing, Jena)"),
       ("Fair comparison with FPRM & existing methods", "noteval", "원본 FPRM은 참조 필요. 조건 B에서는 Adapted FPRM만(Phase 1). Lag-FPRM 없음, 포트 결함 미해결")]
FUT = [("결합 인코더 < 병치의 원인 분리", "최적화 step 일치, gradient 결합만 차단, 같은 함수 부류의 사전등록 통제 설계(Phase 8과 9의 공통 부정적 결과)"),
       ("디코더 단사성과 잠재-상태 Lipschitz 측정", "기존 checkpoint로 측정 가능(학습 불필요). A3.2 조건과 연결"),
       ("식별 층에 한정한 상태 일관성 지표의 사전 정의", "비식별 층은 분포 일관성으로 분리"), ("FPRM 공정 비교의 선결 과제", "포트 D6′ 결정, MATLAB/Octave 교차검증, Lag-FPRM 명세, 같은 정보 조건"),
       ("독립 실제 자료 계열 확보", "현재 Jena는 같은 계열의 seed 8개. 독립 계열이 있어야 추론 가능")]
DATA = dict(phases=PH, math=MATH, aim=[dict(c=c, s=s, e=e) for c, s, e in AIM], fut=[dict(t=t, d=d) for t, d in FUT], tests=T, jena=JENA,
            charts=dict(p9imp=CH_P9_IMP, p9ret=CH_P9_RET, p8imp=CH_P8_IMP, p7=CH_P7),
            src=dict(p9imp=L(P9I), p9ret=L(P9R), p8imp=L(P8I), p7=L(P7), jena=L("05_Imputation_Results/Phase9_Jena_real_data/jena_rep_rmse.csv")),
            docs=dict(synth=L("01_Research_Synthesis/phase1_to_phase9_research_synthesis.md"), keyf=L("research_key_findings.md"), master=L("research_experiment_master_table.csv"),
                      plan=L("08_Future_Research/research_continuation_plan.md"), excl=L("EXCLUDED_FILES_INDEX.md"), start=L("00_START_HERE.md"), repro=L("09_Reproducibility/reproduction_guide.md")))
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html")).read().replace("/*__DATA__*/null", json.dumps(DATA, ensure_ascii=False))
open("index.html", "w").write(html)
print("index.html written;", len(T), "tests;", "missing links:", MISSING)
