# Phase 7 Gate P7-B — Pre-registration: Full-target versus Self-target Reconstruction (2026-10-28)

**고정:** 학습·평가 전, 어떤 P7-B 수치도 보기 전에 고정했다. 해시는 `results/extra/phase7/PROTOCOL_FREEZE_p7b.txt`에 있다. 변경은 §9에 기록한다.

**근거 문서**
- `phase7_mathematical_corrections.md`(A4: G3 음성 결과는 자기 목표 AE에 국한되며, raw 우월성은 필요조건이 아님)
- `phase6_g2_g3_results.md` §3

## 1. 중심 질문
- 전체 multi-view 복원 감독(4 view 모두에 손실)이 자기 하위계 복원 감독(자기 2 view에만 손실)보다, 학습된 잠재가 결합 동역학 상태(6차원)의 일관된 근사를 더 잘 보존하게 하는가?
- 이 실험은 **현재 FDR-Learned 구조와 학습 목표에 관한 반증 가능한 비교**다. 모든 공유 상태 모델에 대한 진술이 아니다.

## 2. Arm (`experiments/phase7/core7.py`, `train7.py`)

| Arm | 입력 | Encoder / Fusion / Decoder | Decoder 출력 | 손실 활성 열 |
|---|---|---|---|---|
| Bself1 | x1, z1, x2, z2 (같은 순서) | Phase 5 ArmModel(FDR-Learned), 같은 구조 | 4 view 전부(view마다 지연 8) | x1, z1 |
| Bself2 | 같음 | 같음 | 같음 | x2, z2 |
| Bfull | 같음 | 같음 | 같음 | x1, z1, x2, z2 |

**같은 것**
- 입력과 순서, 파라미터 수(초기화 seed 7000 + r이 같으면 초기 가중치도 비트 단위로 같음)
- 입력 마스크 배열, 학습 숨김 RNG 흐름(텐서 모양이 같음)
- 자료 분할, 정규화(Phase 6 G2 풀 상수)
- 학습 예산: CFG는 Phase 5/6과 같음(epoch ≤ 300, patience 15, 100 step/epoch, batch 64, lr 1e-3)
- 평가 절차

**다른 것 (명시)**
- 손실이 활성화된 목표 좌표만 다르다.
- 따라서 다음이 다르다.
  - **유효 gradient 경로:** Bself_j에서는 비활성 view의 decoder head에 gradient가 정확히 0이다(테스트로 확인).
  - **감독 신호량:** Bfull은 목표 열이 2배다.
  - **조기 종료 기준:** 각 Arm의 손실 활성 열에서 val 관측 항목의 20%를 숨긴 재구성 RMSE.
- 같은 epoch 상한이라도 실제 학습 길이가 다를 수 있다. epoch 수를 보고한다.

**금지 사항 (테스트로 확인)**
- hidden y1, y2는 AE의 입력, 목표, 정규화, 손실에 쓰지 않는다.
- 새 손실이나 하이퍼파라미터 탐색은 하지 않는다.

## 3. 자료와 반복 (`gen7.py`)
- 새 독립 궤적 id 40000+를 쓴다. 이전 모든 Phase의 id·시드와 겹치지 않는다(테스트).
- 생성기와 시드 규칙은 gen6와 같다(결합 6차원 적분).
- 복제본 r = 0..23 / (κ, p). 복제본마다 train 1(20,000), val 1(5,000), test 8(각 5,000)이 모두 독립이다.
  - **복제본이 추론 단위다(서로 완전히 독립).**
- 조건: κ ∈ {0, 1} × p ∈ {0.3, 0.7}. 학습 4 × 24 × 3 = **288개**.
- 마스크 seed: `seed_of("p7train"|"p7val"|"p7valhide", κ, p, r)`, test `seed_of("p7test", κ, p, r, i, e)`. 초기화 seed는 7000 + r이다.
- 반복 수 근거: 복제본 n = 24이면 표준화 효과 d_z = 0.6에서 검정력이 약 0.9(양측 α = 0.05)다. 기대 효과가 이보다 작아도 n을 줄이지 않는다.

## 4. 평가 (`evaluate7.py`, 결과 전 고정)

**공유 readout**
- 특징 집합마다 **하나의** MLP(2 × 128 ReLU, `evaluate6.mlp_fit`, seed 0)를 쓴다.
- 동결 특징에서 정규화된 **전체 6차원 상태**로 사상한다.
- 복제본의 train 궤적(자연 train 마스크)에서 학습하고, val 궤적으로 조기 종료한다.
- 마스크별 readout은 없다.

**특징 집합**
- Bself1, Bself2, Bfull 잠재
- raw_joint(원시 마스크 4-view 지연 창): **참고 기준선**이며, 판정 기준이 아니다(정정 A4).

**평가 대상:** 복제본의 test 궤적 8개 × 독립 마스크 5개(e = 0..4)이고, 같은 실제 상태에 서로 다른 마스크를 적용한다.

**지표**
- 좌표 s6, s1, s2, y1, y2별 MSE / RMSE
- 마스크 쌍 출력 차이: 안정성 SD = RMS‖r(M) − r(M′)‖ / √2
- 층
  - 마스크만: all, both_L2+, one_L2+, none_L2+, L_j ∈ {L0, L1, L2+}
  - Oracle-only: 4 view 행의 국소 Jacobian σ_min 구간 [0, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), [1e-1, ∞), 매 5번째 시각(`jac7.py`)

## 5. 주 결과와 판정 (`analyze7.py`)

**복제본 지표**
- Q_r(F) = RMSE_s6(F) = sqrt(test 궤적 8 × 마스크 5의 합동 MSE / 6), 층 all.
- Q_r(Bself) = ½[Q_r(Bself1) + Q_r(Bself2)].

**주 대비:** Δ_r = Q_r(Bfull) − Q_r(Bself). 95% 양측 paired t-CI(n = 24).

**고정 순서 (FWER ≤ 0.05)**
- **S1 κ = 1, p = 0.7** → S2 κ = 1, p = 0.3 → S3 κ = 0, p = 0.7 → S4 κ = 0, p = 0.3.
- 앞 단계가 **검출(CI 상한 < 0)**되었을 때만 다음 단계를 확증적으로 판정한다. "보류" 범주는 두지 않는다. 검출이 아니면 그 단계에서 멈춘다.

**실질 효과 기준**
- δ₆ = 0.05(정규화 6차원 RMSE 단위)다.
- 각 단계에서 다음을 따로 보고한다: "δ₆ 이상 확증"(CI 상한 < −δ₆), 평균 ≤ −δ₆ 여부, d_z, n−, Wilcoxon p.

**실패 기준**
- 비유한 실행이나 상한 도달은 제외하지 않고 보고한다.
- 어떤 조건에서 상한 도달 > 10%이면 그 조건은 결론 불가다.
- 복제본은 제외하지 않는다.

**보조 (서술적, 다중성 보정 없음)**
1. 하위계별:
   - Bself_j의 자기 s_j와 상대 s_{3−j}
   - Bfull의 같은 s_j, s_{3−j}
   - 대비 "상대 하위계: Bfull − Bself_j"와 "자기 하위계: Bfull − Bself_j"(감독 분산의 비용)
2. 마스크 간 안정성 SD(s6, 하위계별): Bfull − Bself
3. raw_joint 참고값과 Bfull − raw_joint(필요조건 아님)
4. 마스크 층과 Oracle σ_min 층별 Δ
5. κ의 차이(Welch DiD), p의 차이
6. 학습 진단: epoch, best_val

## 6. 해석 규칙 (사전 고정)
- S1이 검출되면 다음과 같이 보고한다: "현재 FDR-Learned 구조에서 전체 view 감독은 자기 view 감독보다 공유 readout의 6차원 복원 오차를 낮춘다."
- 그래도 **다음으로 확장하지 않는다:** 잠재의 전역 단사성, 참 상태공간과의 homeomorphism, diffeomorphic correspondence, 모든 결측 패턴에서의 상태 식별가능성.
- S1이 검출되지 않아도 "모든 공유 상태 모델이 실패한다"고 주장하지 않는다.
- **"현재 모델이 전체 상태 정보를 보존하도록 학습되는가"의 판단 근거:** 보조 1–3을 함께 본다.
  - Bfull의 상대 하위계 오차가 Bself보다 작고 raw_joint 참고값에 가까우면 → Bself의 정보 손실은 **학습 목표**로 설명된다 [Empirical].
  - Bfull도 raw_joint보다 크게 나쁘면 → 목표만으로는 설명되지 않는다. 다른 원인(잠재 16차원 병목, 마스크 의존 변동, 최적화)은 Hypothesis로 남긴다.
  - DPI 때문에 Bayes 수준에서 잠재 readout이 raw를 이길 수 없다는 점을 함께 적는다.

## 7. 실행
- GPU: 사용자 예약(fai-L40S GPU 0, 1) 안에서 `CUDA_DEVICE_ORDER=PCI_BUS_ID`로 실행한다. lane에는 마감 guard를 둔다.
- 마감 전에 끝나지 않은 셀은 "미실행"으로 보고하고, 완료된 셀만으로 판정하지 않는다(해당 단계는 결론 불가).

## 9. 변경 기록
- (없음)
