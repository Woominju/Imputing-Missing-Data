# Phase 1–9 통합 연구 요약

- **과제:** Imputing Missing Data in Complex Dynamical Systems using Delay Embedding Theorem
- **멘토:** Dongju Lim
- **작성:** 2026-10-31(Claude Code)
- **경로 규칙:** 모든 경로는 프로젝트 루트 `/home/minju/faicpaston/Imputing_Missing_Data` 기준이다. 아카이브 안에서는 같은 파일이 각 폴더에 사본으로 들어 있다.

**라벨**
- **[증명]** 수학적 증명이 있다(Verified (math)).
- **[인증]** 수학 명제 + 통계적 하한/상한으로 인증했다.
- **[확증]** 사전등록 확증 검정을 통과했다.
- **[서술]** 경험적 관찰이며 확증이 아니다.
- **[가설]** / **[미해결]**
- **[무효]** 이후 정정됐다.

---

## 1. Official Research AIM (원문, 변경 금지)

> We aim to develop a dynamics-informed imputation framework that reconstructs missing values in multivariate time series without requiring any fully observed variable. It does so by learning a shared latent state to which the delay embeddings of all variables are mapped consistently, as implied by Takens' embedding theorem.

## 2. 연구 배경

**지연 매장 (Takens 1981; Sauer–Yorke–Casdagli 1991; Deyle–Sugihara 2011)**
- 어떤 관측 함수와 지연 수에서 지연 좌표 사상은 상태공간(또는 끌개) 위의 임베딩이 된다. 단, 이 결과는 **generic/prevalent** 관측에 대한 것이다.
- 우리 문제는 다음 두 가지 이유로 이 정리를 직접 적용할 수 없다.
  - (i) 관측 함수가 고정된 좌표 사영이다(예: Lorenz의 x, z).
  - (ii) 결측 때문에 지연 벡터의 일부 좌표가 지워진다.
- 출처: `results/notes/math_review.md` §1–3, `phase9_mathematical_foundations.md` §A5

**FPRM** (Nat. Commun. 2026, DOI 10.1038/s41467-026-77922-1)
- 가우스 과정으로 **완전 관측된 참조 변수 하나**의 지연 벡터에서 다른 변수를 복원한다.
- AIM은 바로 이 참조 변수 가정을 제거하려 한다.
- 출처: `results/notes/fprm_comparison_protocol.md`, `fprm_python_port_report.md`

**Multi-view AE**
- 변수(view)마다 결측이 있는 과거 지연창을 인코딩하고, 공유 잠재에서 각 view의 지연 벡터를 복원한다.
- 학습 목표는 숨긴 관측 항목뿐이다(완전 관측 참조 불필요).

**실험 시스템**
- Lorenz, 그리고 결합 Lorenz다. 결합 Lorenz는 하위계 2개이고 결합 κ ∈ {0, 1}이다.
- 관측은 x1, z1, x2, z2이고 y1, y2는 숨겨진다.
- 결측은 MCAR이며 p ∈ {0.3, 0.7}이 주 조건이다.
- 실제 자료는 EEG, ENSO, Exchange, Wind, Transportation(FPRM 논문 자료), 그리고 Jena와 Beijing이다.

## 3. 연구 연표

| Phase | 날짜 | 핵심 질문 | 핵심 결론 |
|---|---|---|---|
| 1 | 10-24 → 10-27 | 다중 view 지연 AE가 모든 변수 MCAR에서 대치를 할 수 있는가? FPRM 전제는 성립하는가? | 모델 C가 조건 B(참조 없음)에서 작동한다. z 참조 FPRM은 대칭 때문에 실패한다(nRMSE 1.006–1.069). 공유 잠재 일관성은 미달이다 |
| 2 / 2.5 | 10-27 | 부분 관측 식별성의 수학 정리와 구조 설계 | 무조건 점 일관성은 정보를 붕괴시킨다(P7). PoE 정확성 주장은 철회됐다. FDR을 먼저 구현하기로 결정했다 |
| 3A–3D | 10-27 | FDR(전체 지연 재구성)이 숨은 상태 정보를 늘리는가? | 숨은 y 정보는 늘지만 대치는 나빠진다(trade-off). 3C의 "일관성 개선"은 **평가 오류로 무효**(3D-1)다 |
| 4 | 10-27 | 결합 Lorenz에서 다중 view 결합이 식별과 복원을 돕는가? | p에 따라 decoder 효과의 부호가 바뀐다. view 제거 비교는 분포 밖이었다(대체됨). 단일 궤적이다 |
| 5 | 10-28 | 상대 view 입력(Arm B)이 자기 숨은 y를 개선하는가? | H2는 지지되지 않았다(B−A +0.0137). 입력 비용이 이득보다 크다. Sham CI는 무효(E6)다 |
| 6 | 10-28 | 정보 존재(H1), 사용(H2), 좌표(H3)의 분리 | 대칭 기반 정보 하한을 얻었다. B−Sham −0.0070. **κ=0 음성 대조 실패**(미해결) |
| 7 | 10-28/29 | 학습 목표(B-full 대 B-self)가 잠재의 6차원 상태 보존을 바꾸는가? | RMSE_s6 −0.37(24/24). κ=0에서도 같으므로 결합 활용이 아니라 **감독 범위**의 효과다 |
| 8 | 10-29 → 10-31 | 정보 한계 대 모델 한계, 결합 특이 정보 접근(CSIA), 결합 인코더 대 병치 | 모델 한계 ≥ 0.041 MSE, CSIA 확증(작음). **B-full > Fact8은 미확증**이며 서술적으로는 반대다 |
| 9 | 10-31 | 수학적으로 정당화된 단일 메커니즘(tower 자기 증류)이 공유 상태 학습과 대치를 개선하는가? | **P-1 실패**(대치 +0.0021, 반대 방향). 공유 상태 개선은 미미하다. Phase 8의 부정적 결과가 재현됐다 |

## 4. 수학적 결과

### 4.1 증명된 정리와 반례 [증명]
1. **Lorenz 대칭군**(Phase 5 Lemma S, Phase 8 L1): Γ(x, y, z) = (−x, −y, z)
   - 동시 반전 γ₁₂는 모든 κ에서 흐름과 가환한다.
   - 개별 반전 γ_j는 κ = 0에서만 가환한다.
   - 초기분포와 MCAR 마스크까지 포함한 결합 법칙이 이 군에 불변이다.
   - 출처: `phase8_mathematical_protocol_v2.md` §1.3
2. **z만 쓰는 지연 사상은 단사가 아니다**(2:1 이상): `math_review.md` §2(b), Phase 6 정오표 E1("정확히 2:1" → "적어도 2:1")
3. **X0 대칭 하한 N1-X0:** 창 안에서 x1, x2가 모두 결측이면 E[x, y | W] = 0이고, R*_X0 ≥ (1/6)Σ_{c ∈ {x,y}} E s_c²/σ_c²다.
   - 이 값은 분산 일치일 때만 정확히 2/3이다.
   - 실측 LB는 κ=1에서 0.6679 [0.6670, 0.6687], κ=0에서 0.6667이다.
   - 출처: protocol v2 §3, `phase8_postrun_independent_audit_20261031.md` §A2
4. **κ=0 인수분해 P8.1:** (s₁, W₁) ⊥ (s₂, W₂)이므로 교차 정보의 Bayes 값은 0이다.
5. **Bayes 위험 부등식 B1:** 고정된 f에 대해 R(h) − R* ≥ R(h) − R(f)다. **B3:** 잠재 readout과 raw readout의 차이는 인코더의 Bayes 정보 손실이 아니다.
6. **P8.4 인증 정리:** L_j0에서 C_{F,j} > 0이면 F(W)는 W_j의 함수가 아니다.
7. **반례 C1–C5**(protocol v2 §3)
   - L_j ≥ 2가 식별을 보장하지 않는다.
   - 국소 rank는 전역 단사가 아니다.
   - 좋은 대치는 숨은 상태 복원이 아니다.
   - 안정성은 식별이 아니다.
   - 주변 충분성은 결합 충분성이 아니다.
8. **Phase 9 Gate A** (`phase9_mathematical_foundations.md`)
   - **A2.2:** 무조건 cross-mask 잠재 정렬은 상태 구별성과 양립하지 않는다.
   - **A3.1:** 정확한 등식 E_m∘π_mΦ = T, D∘T = Φ가 성립하면 T는 C¹ 임베딩이다. 이는 식별 마스크에서만 가능하다.
   - **A3.2:** 근사 단사성
   - **A4.1–A4.2:** MCAR, holdout, 양성 조건에서 관측 항목 손실은 조건부 기댓값에 Fisher 일관이다. **완전 관측 참조 없이 학습하는 것의 수학적 근거**다.
   - **A4.3:** tower 자기 증류 목적은 유일한 고정점을 가진다. 참값 항목이 없으면 붕괴한다.
9. **Phase 2의 P7**(점 일관성은 정보 붕괴): `aim_mathematical_foundation.md:186–190`. A2.2의 전신이다.

### 4.2 통계적 인증 [인증]
- **Phase 6 G1:** Δ*_L0의 LCB는 0.0322이고, Δ* ≥ p⁸ × 0.032 = 0.0019다(κ=1, p=0.7; `phase6_g1_results.md`).
  - Phase 7 정정에 따라 "Verified (math)"에서 **Empirical(근사 bootstrap LCB)**로 강등됐다(`phase7_mathematical_corrections.md` §A5).
- **Phase 8 I-1:** H층에서 모델 한계는 ≥ 0.041 MSE다(97.5% LCB, 풀 조건부).
- **Phase 8 IIa-2:** κ=1 L_j0에서 φ_Bfull(W)는 W_j의 함수가 아니다(C_Bfull LCB 0.0245).
- **Phase 8 X0:** B-full 위험의 83% 이상은 정보 한계다.

### 4.3 철회와 정정된 수학적 주장 [무효]
- MCAR ⇒ 가우스 PoE 정확(P9)은 철회됐다.
- "ELBO 최적 ⇒ 참 사후분포"는 틀렸다(조건부 ELBO가 아님). 출처: `phase25_mathematical_audit.md`, `phase25_objective_redesign.md`
- Phase 6에서 다음 세 주장이 하향 조정됐다(`phase7_mathematical_corrections.md` §A5).
  - Δ* > 0 수학 검증
  - "S1R이 S1을 독립 확증"
  - "S3 확증"

## 5. 실험 결과 (핵심 수치; 단위와 출처 명시)

| # | Phase | 대비 (조건) | 결과 | 단위 / 판정 | 출처 |
|---|---|---|---|---|---|
| 1 | 1 | 조건 B(참조 없음, Lorenz dt=0.05), C 대 linear | C 0.0091/0.0179/0.0387/0.1424 대 0.1472/0.2130/0.3517/0.5713 (p=0.1/0.3/0.5/0.7) | 정규화 RMSE, 3 seed 평균(단일 궤적) [서술] | `aim_autonomous_20261027.md:100–120` |
| 2 | 1 | z 참조 FPRM-T (조건 A) | nRMSE 1.006–1.069 | 대칭 실패를 예측과 일치하게 확인 [서술] | 같은 문서 :122–135 |
| 3 | 3D-2 | FDR(α=1/8) − C | 대치 5/5 seed에서 악화, y-probe 5/5 개선 | 단일 궤적, seed 5개 [서술] | `phase3d2_fdr_lag_weight_results.md:55–74` |
| 4 | 5 | B − A (κ=1, p=0.7, y_j RMSE) | +0.0137 [−0.0015, +0.0288] | 복제본 n=8, H2 미지지 | `results/extra/phase5/analysis/contrasts_probe_mlp.csv` |
| 5 | 6 | B − Sham S1 (κ=1, p=0.7) | −0.0070 [−0.0116, −0.0024], 19/24 | n=24. **κ=0 대조 +0.0062 [0.0022, 0.0101] 실패** | `results/extra/phase6/analysis/g2_fixed_sequence.csv`, `g2_contrasts.csv` |
| 6 | 6 | S1R B − Sham (κ=1) | −0.0104 [−0.0141, −0.0068], 23/24. κ=0은 +0.0009(TOST 동등) | n=24. 사후 설계 재현 | `results/extra/phase6/s1r/s1r_results.csv` |
| 7 | 7 | Bfull − Bself, RMSE_s6 (S1, κ=1, p=0.7) | −0.3702 [−0.3755, −0.3649], 24/24. S2–S4도 모두 확증(κ=0 S3 −0.4092) | n=24 [확증]. 결합 효과 아님 | `results/extra/phase7/analysis/p7_fixed_sequence.csv` |
| 8 | 8 | I-1 G_tot (κ=1, p=0.7, H) | 0.04412, LCB 0.04117 > 0.02 | MSE, n=24 [확증·인증] | `p8_familyI_sequence.csv` |
| 9 | 8 | I-2 G_lat | 0.01898 [0.01746, 0.02050] | 실패(δ 근방) | 같음 |
| 10 | 8 | IIa-1 CG_Bfull (Real − Sham) | −0.00863, UCB −0.00548, 21/24 | y RMSE [확증], δ_y=0.02 미달 | `p8_familyII_sequence.csv` |
| 11 | 8 | IIb-1 Δ_JF (B-full − Fact8 결합 이득) | −0.00278, UCB +0.0000286 | 미검출(경계) | 같음 |
| 12 | 8 | RMSE_s6(Bfull) − (Fact8) | +0.0176 [0.0124, 0.0229] | [서술] B-full이 나쁨 | 같음 |
| 13 | 8 | 대치 RMSE (κ=1, p=0.7) | Bfull 0.3121, Fact8 0.2959, Fact16 0.2932, LOCF 0.9303, O_imp(지도) 0.2595 | 결측 위치 4 view 합동, n=24 [서술] | `results/audit/phase8_postrun_20261031/out/indep_familyIII_summary.csv` |
| 14 | 9 | P-1 대치 P9 − Bfull | +0.00211 [LCB +0.00006, UCB +0.00416] | **확증 실패(반대 방향)** | `results/extra/phase9/analysis/main_20261031-213423/p9_confirmatory_sequence.csv` |
| 15 | 9 | 대치 P9 − Fact8 / ρ_ret P9 − Bfull | +0.0197 / −0.0040 | [서술] 비열등 실패 방향, 검색 개선 미미 | 같음, `p9_paired_all_cells.csv` |
| 16 | 9 | Jena 대치 (p=0.3/0.7) | Bfull 0.290/0.454, P9 0.294/0.459, SelfV 0.330/0.530, LOCF 0.348/0.524 | 같은 계열에서 seed 8개 [서술] | `results/extra/phase9/jena/jena_rep_rmse.csv` |

## 6. 부정적 결과와 교훈

**기존 접근이 공유 상태를 만들지 못한 이유**
- **무조건 정렬은 원리적으로 불가능하다.** 정렬을 강제하면 비식별 마스크의 대칭 궤도를 붕괴시킨다(P7, A2.2).
- **view dropout은 단일 view 일관성을 개선했지만 다변량 정확도를 해쳤다**(Phase 1–3A).
- **Phase 9 tower 자기 증류는 추정 대상을 바꾸지 않는다**(A4.3). 따라서 유한 용량에서의 이득에만 기댈 수 있었는데, 그 이득이 나타나지 않았다. p=0.3에서는 악화했고, 3/24 run에서 학습이 발산했다.

**존재하지만 활용되지 못한 정보**
- 상대 view에는 L_j0층에서 자기 숨은 y의 정보가 있다(Phase 6 G1, Phase 8 IIa-2).
- 그러나 파이프라인이 실현하는 이득은 작다. S1R −0.0104, IIa-1 −0.0086으로 둘 다 δ = 0.02보다 작다.
- Phase 7에서 공유 readout은 L0층 y_j를 영 예측보다 못하게 예측했다(1.019).

**성립하지 않은 가정**
- "L_j ≥ 2이면 식별 가능"(C1에서 반례)
- "MCAR이면 PoE가 정확"(철회)
- "FDR이 일관성을 개선"(3C 무효)
- "B-full의 이득이 결합 활용"(Phase 7, κ=0에서도 같음)

**상태 식별이 불가능한 마스크**
- X0(창 안에 x가 없음), z만 있는 창, 완전 결측 창, n_obs < 6(P에서)
- Phase 9 X0 질의의 cross-mask 검색은 무작위 수준이었다(0.80–0.84).

**결합 인코더가 병치보다 낫지 않은 이유 [미해결]**
- Phase 8 IIb-1은 미검출이었다.
- 6차원 decodability(+0.018)와 p=0.7 대치에서 Fact8이 낫다. 이 결과는 Phase 9의 독립 자료에서도 재현됐다(+0.018, +0.031).
- 원인 후보는 함수 부류, 하위계 간 gradient 결합, Fact의 2배 최적화 step, 결측률에 따른 귀납 편향인데, **분리되지 않았다.**

**통계 설계의 교훈**
- 단일 궤적 seed는 독립 복제가 아니다(Phase 1–4).
- donor 재사용은 Sham CI를 무효로 만든다(E6).
- 평가 지표에 관측 좌표를 섞으면 안 된다(3D-1).
- 고정 후 변경은 이중으로 기록해야 한다(Phase 8 F3).

**재현에서 남은 결함**
- Phase 6 κ=0 대조 실패의 원인은 미해결이다.
- FPRM Python 포트에는 국소 정체(D6′) 결함이 있다.

## 7. 제안된 방법론 (현재 시점)

**기본 틀:** mask-aware multi-view ArmModel
- view별 인코더, 학습 fusion, 공유 잠재, view별 전체 지연 재구성 디코더로 이루어진다(FDR).
- 손실은 숨긴 관측 항목에만 준다(완전 관측 참조 불필요, A4.1).
- 코드: `experiments/phase5/core.py`, `experiments/fdr/loss.py`

**감독 범위:** 모든 view를 손실 열로 쓰는 B-full이 기본이다(Phase 7).

**Phase 9 P9-MVAE:** B-full에 tower 자기 증류를 더한 것이다. 수학적으로 정당하고(추정 대상을 보존) 붕괴하지 않는다. **그러나 경험적 이득이 없어서 기본 방법으로 채택하지 않는다.**

**경험적으로 가장 강한 비교 기준:** 하위계별 병치 인코더(Fact8, Fact16)다. p=0.7 대치와 6차원 decodability에서 우위다.
- 이는 **AIM이 말하는 "하나의 공유 잠재"가 아직 병치보다 낫다는 증거가 없다**는 뜻이다.

## 8. Official AIM 정합성

| 구성 요소 | 상태 | 근거 |
|---|---|---|
| 완전 관측 변수 없이 대치 | **Established** | A4.1(수학), 모든 모델이 참조 없이 학습(코드). 대치 RMSE가 LOCF와 zero보다 크게 낮음 |
| 다중 view 지연 매장 통합 | **Partially Supported** | 상대 view 정보 존재(인증)와 활용(IIa 확증, 작음). Jena에서 다중 view가 변수별 AE보다 나음(서술). 결합 인코더 우위는 없음 |
| 일관된 공유 잠재 상태 | **Unresolved** | G3·GSR 기준 미달. P9 개입 효과 미미. 비식별 마스크의 원리적 한계(A2) |
| Takens형 수학적 대응·식별성 | **Unresolved** | 조건부 정리(A3.1)만 있음. 학습 모델이 가정을 만족한다는 증거 없음 |
| 시뮬레이션과 실제 자료 검증 | **Partially Supported** | 결합 Lorenz는 사전등록 확증 설계. 실제 자료는 서술(Phase 1 FPRM 자료 5종, Phase 5 Beijing/Jena, Phase 9 Jena) |
| FPRM 및 기존 방법과의 공정 비교 | **Not Evaluated**(공정 비교 기준) | FPRM은 참조 변수를 요구함. 조건 B에서는 Adapted FPRM만 실행됨(Phase 1). Lag-FPRM 미구현, 포트 결함 미해결 |

## 9. 열린 수학 문제
1. 고정 좌표 관측(x, z)과 결측 패턴에서 지연 사상의 임베딩성(A3 (H2))과 Lipschitz 역
2. H∖X0 층의 정량적 Bayes 하한(현재는 X0와 κ=0 L_j0만 있음)
3. 출력 공간 일관성에서 잠재 상태 일관성으로 가는 조건(디코더 국소 단사성)
4. 유한 모델에서 결합 인코더가 병치에 지는 원인의 수학적·실험적 분리
5. ν(불변측도)에서의 식별 문제(현재 결과는 모두 유한 시간 법칙 P 기준)
6. 반-경사 자기 증류의 유한 모형 안정성
7. Phase 6 κ=0 음성 대조 실패의 원인
8. ½·1-NN 통계량(D5b)의 일관성 조건(유한 표본에서 하한 아래로 나옴)

## 10. 향후 연구
상세 계획은 아카이브 `08_Future_Research/research_continuation_plan.md`에 있다.

**최우선 질문:** "왜 결합 인코더가 병치보다 낫지 않은가?" 이를 원인별 통제 설계로 분리하는 것이다.
- 최적화 step 일치
- gradient 결합만 다른 arm
- 같은 함수 부류

**그 밖의 과제**
- 디코더 단사성과 Lipschitz 역의 측정
- FPRM과의 정보 조건 일치 비교
- 실제 자료의 독립 계열 확보
