# Phase 8 Gate B — Protocol Integrity Report (2026-10-29)

**상태:** Gate A(수학 개정)와 Gate B(무결성)는 완료했다. **Gate C(해시 고정)와 Gate D(GPU 실행)는 진행하지 않았다.** 사용자 결정을 기다린다(§5).

**대상**
- 프로토콜: `phase8_mathematical_protocol_v2.md`(632줄, md5 `74f290c4…`)
- 대조표: `phase8_revision_crosswalk.md`
- v1 원문: md5 `e8bb1491…`, SHA-256 `c5e37088…`(보존)
- 코드: `experiments/phase8/`(약 1,300줄)

**실행하지 않은 것**
- GPU를 쓴 작업은 하나도 없다. 이 시점에 nvidia-smi compute 프로세스는 0개다.
- Phase 8 실제 자료 생성, AE·oracle 학습, 평가는 하지 않았다. `results/extra/phase8/`는 존재하지 않는다.
- 모든 검증은 CPU smoke(임시 디렉터리, 축소 크기)로 했다.

## 1. 최종 확정 항목 (v2 기준; 고정 전)

| 항목 | 확정 내용 | v2 위치 |
|---|---|---|
| Primary (Family I) | κ = 1, p = 0.7, H에서 G_tot = R_H(R1∘φ_Bfull) − R_H(O_raw), LCB_{0.975} > δ_I = 0.02(MSE). 고정 순서 I-1 → I-2(G_lat) → I-3(κ = 0 G_tot) → I-4(κ = 0 G_lat) | §4.5 |
| Primary (Family IIa) | κ = 1, p = 0.7, R1, CG_Bfull = ½Σ_j[RMSE(y_j \| Bfull) − RMSE(y_j \| BfullSham_j)]. UCB_{0.9875} < 0 → IIa-2: P8.4 인증 C_Bfull(O_lat), LCB > 0 | §5.3 |
| Primary (Family IIb) | Δ_JF = CG_Bfull − CG_Fact8, UCB < 0 → IIb-2: RMSE_s6(Bfull − Fact8) UCB < 0 | §5.3 |
| 추론 단위 | 독립 복제본 24 / (κ, p). 궤적·마스크·j는 복제본 안에서 합동한다. oracle은 풀에 조건부다. κ는 독립(Welch), p는 대응 | §9 |
| 대조군 | κ = 0 Real 대 Sham(j별 정확 대칭), κ = 0 P8.4(C ≤ 0), BfullSham P8.4, O_raw / f_kNN 상한, X0와 κ = 0 L_j0 하한, κ = 0 Bfull 대 Fact8(병치 비용) | §8 |
| 동등성 마진 | κ = 0 게이트: j별 정확 부호 검정(각 0.025, FAIL) + TOST ε = 0.01(PASS). 그 외는 UNDETERMINED(확증 보류 라벨) | §5.3 |
| 다중성 | Family I 0.025 + IIa 0.0125 + IIb 0.0125 = FWER 0.05(단측), 각 고정 순서 | §9 |
| 검정력 | n = 24에서 80% 검정력의 최소 d_z = 0.60(α 0.025) / 0.67(α 0.0125). Phase 8 대비의 d_z는 미지다. 미검출은 "이 n에서 미검출"로 보고한다 | §5.4 |
| 실질 효과 | δ_I = 0.02(MSE), δ_y = 0.02, δ₆ = 0.05(RMSE). 방향성(평균 ≤ −δ)으로 따로 보고한다 | §4.5, §5.3 |
| 실패 처리 | 셀의 상한 도달 > 10%면 결론 불가. 미실행 셀을 보고한다. smoke 산출물은 분석이 거부한다 | §9 |

## 2. 코드 ↔ 프로토콜 대응 (확인 방법)

| 프로토콜 요구 | 코드 | 확인 |
|---|---|---|
| id·seed 비중첩(P5–P7, 복제본 / donor / 풀 / 참조) | `core8.ids`, `pool_ids`, `ref_ids` | test 1(전체 크기) |
| 상수가 문서 수치와 같음 | `core8.CONST`, `FULL` | test 2(문서 문자열 31개 대조) |
| Arm 표(입력 view, real/donor, 잠재, seed, 파라미터) | `core8.arm_spec`, `train8.py` | test 3, smoke(파라미터 57,840 / 27,944) |
| 포획 항등식, V ≤ 3850.667, 3080.533 | — | test 4(sympy 항등식, 최악 모서리 궤적) |
| 대칭군: γ₁₂와 π는 모든 κ, γ₁은 κ = 0에서만 | — | test 5(수치 적분. κ = 1에서 γ₁ 위반을 확인) |
| X0·L_j0에서 창 불변(N1의 전제) | `core8.strata`, raw_features | test 6 |
| 층 확률, X0/H = 0.0509 | `core8.strata` | test 7(Monte Carlo) |
| T_A의 1/6 정규화, sym0 = −μ/σ | `core8.sym0_pred`, `analyze8` | test 8 |
| 통계 함수, 게이트(PASS / FAIL / UNDETERMINED) | `analyze8.bounds`, `sign_test`, `tost`, `gate` | test 9 |
| R1 = P7 readout 규칙 | `core8.fit_mlp` | test 10(`evaluate6.mlp_fit`과 비트 단위 동일) |
| past-only | ArmModel / encode_series | test 11 |
| k-NN 창 배치, 동률 처리 | `knn8.windows`, `knn_predict` | test 12 |
| D5b 정확 조건부 마스크 추출 | `knn8.sample_window_mask` | test 13(기각 추출과 분포 일치, P(X0) = 4e-9에서도 정확) |
| 전체 작업 계획 1,277개(864 / 192 / 96 / 96 …), dry-run 부작용 없음 | `sched8.tasks` | test 14 |
| P7 대치 평가 경로 | `eval_p7imp8.py` | test 15(**무작위 가중치 mock**. P7 결과를 고정 전에 계산하지 않음. Bself_j는 자기 view만) |
| smoke는 GPU를 쓰지 않음 | `core8.device`, `sched8` | test 16 |
| 전체 파이프라인 + 데이터 의존 검사 | 전체 | test 17(아래) |
| 이전 Phase 파일 불변 | — | test 18(P7 동결 md5 17개, v1 해시) |

**test 17 (CPU smoke end-to-end, 축소 크기: 복제본 2, test 2 × 마스크 2, 참조 150)**
- gen8 → 작업 125개(학습 72, oracle 36, k-NN / D5b 8, 평가 8, 분석 1)를 실행했다. **실패 0개.**
- 분석 산출 CSV 18종이 모두 생성됐다.
- smoke epoch 상한 때문에 모든 셀이 "결론 불가"로 표시됐다. **상한 > 10% 규칙이 작동함을 확인**했다.
- 데이터 의존 검사
  - hidden y가 입력에 없다.
  - 참조 창 모양과 시각 seed가 맞다.
  - **Sham 교환 검사:** donor := real로 두면 BfullSham_j는 Bfull과, A_j_8d는 A_j_8과 배열 단위로 같다. S^♯ = S다.
  - oracle의 `pool_arrays`가 읽는 id는 풀 id뿐이다.
  - O_imp의 손실 가중치 = 1 − M(현재 결측만).
  - smoke가 아닌 환경에서 analyze8이 smoke 산출물을 **거부(REFUSED)**한다.

**결과:** `phase8_gateB_test_log.txt`. **18/18 PASS**(최종 단일 실행, rc = 0).

## 3. Gate B 중 발견해 고친 결함 (고정 전)
1. **[안전] smoke 모드의 GPU 사용 가능성:** sched8이 smoke에서도 `CUDA_VISIBLE_DEVICES = 0/1`을 지정할 수 있었다. 수정 후 smoke는 강제로 CPU를 쓴다(test 16).
2. **[정확성] D5b 기각 추출:** p = 0.3의 X0(4e-9)에서 끝나지 않는다. 정확 조건부 추출로 바꿨다(test 13). v2 §2.1에 반영했다.
3. **[정확성] k-NN 위험 합산:** n을 `first`로 잡아 위험을 잘못 계산했다. 궤적 × 마스크 × 좌표의 합으로 고쳤다.
4. **[정확성] 실질 효과 판정:** |평균| ≥ δ를 쓰고 있었다. 단측 방향(평균 ≤ −δ)으로 고쳤다.
5. **[무결성] `sched8 --dry`가 빈 로그 디렉터리를 만들었다.** 부작용을 제거했다. 생긴 빈 디렉터리는 비어 있음을 확인한 뒤 삭제했다(test 14).
6. **[설계] Family I 보조 정확 부호 검정이 누락됐다.** 추가했다(G − δ_I 기준).

## 4. 남은 위험과 한계 (차단 여부)

| # | 내용 | 차단 여부 |
|---|---|---|
| B1 | ChatGPT 검토 **원문**을 받지 못했다. 대조는 사용자 요약 12항목 기준이다 | 비차단(원문 수신 시 재대조 권장) |
| B2 | ½·1-NN 일관성 인용 미검증 | 비차단(D5b는 판정에 쓰지 않음) |
| B3 | O_lat 폭을 O_raw 선택 폭으로 고정한 것은 설계 선택이다. G_lat 해석은 "학습된 두 함수의 차이"로 한정된다 | 비차단(사전 명시) |
| B4 | κ = 0 게이트가 UNDETERMINED가 될 가능성. 복제본 SD가 0.02 이상이면 TOST 검정력은 0.54 이하다. Phase 6에서 G2 κ = 0 대조 실패 전례도 있다 | 비차단(처리 규칙을 사전 고정). 단, 확증 결론이 보류될 위험이 있다 |
| B5 | H층과 δ_I는 P7 결과를 보고 정했다(가설 생성). Phase 8은 새 독립 자료로 확증한다 | 비차단(명시) |
| B6 | R1 readout의 학습 자료(train 궤적 20,000 step) 시각 분포가 P(n < 5,000)와 약간 다르다 | 비차단(위험의 정의는 P 기준) |
| B7 | 계산 예산 추정(설계 단계): AE 864개 약 4–5시간(2 GPU), oracle 16 + O_lat 192 + 평가 수 시간, k-NN 참조 생성 약 30분(CPU) | 비차단 |
| B8 | **GPU 예약이 없다**(10/29 08:39 만료) | **Gate D 차단:** 실행 전에 예약 필요 |

## 5. 다음 단계 (사용자 결정 필요)
- **Gate C (고정):** 수학적·실험적 차단 이슈는 없다(B8은 실행 조건). 고정하려면 v2, `experiments/phase8/*.py`, 테스트, 의존 파일의 해시를 `results/extra/phase8/PROTOCOL_FREEZE_p8.txt`에 기록한다.
- **Gate D:** 고정 후, `labres-check`로 예약을 확인하고 사용자가 GPU를 예약한 다음에만 실행한다.
  - 명령: `DEADLINE_EPOCH=… GPUS=0,1 python experiments/phase8/sched8.py NPAR`. 자료 생성 `gen8.py 60`이 먼저다.

## 6. 현재 해시 (고정 기록 아님; 참고)
```
806c8076410448d4ef1efc5d839fd73f  experiments/phase8/analyze8.py
ff755622f5c884340dfa626a8239afda  experiments/phase8/core8.py
64ad5c31e8315086afcc4ab8861a951e  experiments/phase8/eval_p7imp8.py
19220f5896b6b03cd6c7995e97942aab  experiments/phase8/evaluate8.py
0f51e3c1fbf32686ad816d726a407936  experiments/phase8/feats8.py
c953cbb3c6397d2aafe9453eecf4aa7b  experiments/phase8/gen8.py
6976d98de7a940da8f66c65aebcdeadf  experiments/phase8/knn8.py
de9a1257c5951bd18dba41c3f26a8b11  experiments/phase8/oracle8.py
6eabbda19d383e3b9014e42d85f41b0f  experiments/phase8/sched8.py
ffcf36d00f3cc69d1ebf51d6b07a2abb  experiments/phase8/train8.py
c1dafef4a855d2eea9e30e41d80a2f93  experiments/phase8/tests/test_phase8.py
74f290c4e1408c449ee0235102a682ac  results/notes/phase8_mathematical_protocol_v2.md
e8bb14911e6fc2dbca90e451513d9f84  results/notes/phase8_mathematical_protocol.md   (v1, 불변)
```
