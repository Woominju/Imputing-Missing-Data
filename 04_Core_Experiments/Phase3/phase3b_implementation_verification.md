# Phase 3B — Implementation Verification (2026-10-27)

태그: **[Verified]** 테스트나 정확한 재현으로 확인 / **[Empirical]** 측정값 / **[Hypothesis]** 가설 / **[Unresolved]** 미해결.

## 1. 원자료 재집계 (Phase 3A)

`patterns.csv`(168행), `pairs.csv`(156행), `near_collisions.csv`(12행), `pilot/pilot_runs.csv`(4행)를 직접 다시 읽었다. 보고서 값과 일치한다. **[Verified]**

| 항목 | 원자료 재집계 (seed 평균, 범위) | 보고서 값 |
|---|---|---|
| C p=0.3 NATe1\|NATe2 Case A strict / loose | 0.312 (0.214–0.446) / 0.080 (0.039–0.146) | 31.1% / 8.0% |
| C p=0.7 ART:xyz vs NAT:NATe1 상태 오차 | 0.220 (0.197–0.262) vs 0.095 (0.090–0.103) | 0.22 vs 0.095 |
| E p=0.3 ART:x | 0.132 (0.117–0.145) | 0.13 |
| `inv_lip_lower_bound` (C, E) | 192–524 | 260–400 (평균 범위) |
| 파일럿 test RMSE C16 / C3 / E16 / E3 | 0.1157 / 0.1522 / 0.1166 / 0.1510 | 같음 |

## 2. 수학적·방법론적 정정

### 2.1 역-Lipschitz 진단

P11′의 결론 ‖z_B − z_B′‖ ≤ 2Lε에는 **L의 상한**이 필요하다. 즉 해당 영역의 모든 z, z′에 대해 ‖z − z′‖ ≤ L‖R(z) − R(z′)‖이어야 한다.

| 구분 | 내용 | 이번 단계의 측정 |
|---|---|---|
| A. 경험적 거리 비율 d_z/d_R | 표본 쌍의 비율. 그 최댓값은 필요한 L의 **하한** | 측정함 (안정화 비율 d_z / max(d_R, 0.02)의 분위수) |
| B. 국소 Jacobian | 국소 역-Lipschitz만 줌. 전역 아님 | Phase 3A에서 C/E에 대해 측정 (이번 FDR은 미측정) |
| C. 표본 near-collision | 출력이 가깝고 잠재가 먼 표본 쌍 | 측정함 (참 상태 거리 포함) |
| D. 증명된 역-Lipschitz 상한 | 이론적 보장에 필요 | **없음 [Unresolved]** |

Phase 3A의 `inv_lip_lower_bound`는 A의 최댓값이다. 이를 L의 상한처럼 쓴 해석은 철회한다. FDR이 P11′을 만족한다는 주장도 하지 않는다.

### 2.2 행렬 융합 (P8″)

- Λ_i = D_i + U_iU_iᵀ에서 D_i의 대각을 softplus로 양수로 두면 Λ_i는 양정치다. 그러면 **비자명한 영공간이 없다.**
- P8″(존재 증명)는 Λ_z(o_z)의 정확한 영공간이 ψ(s) − ψ(Γs) 방향을 포함한다는 가정을 쓴다. 그래서 이 구현에는 **직접 적용되지 않는다.**
- 작은 양의 고유값 η가 그 방향에 있으면, 정확한 일관성이 아니라 그 방향 기여가 대략 η/(λ_x + η) 배로 줄어드는 근사만 기대할 수 있다. 이 근사식은 유도하지 않았다 **[Hypothesis]**.
- 정확한 0이 필요하면 다음 구조가 필요하다:
  - (a) Λ_i = U_iU_iᵀ (계수 결핍 허용) + 융합 분모 정칙화 εI, 또는
  - (b) 명시적 사영 게이트.
- 그 수학적 정당성을 다시 세운 뒤에 평가한다. **이번 파일럿은 스칼라 융합만 쓴다.**

### 2.3 Lorenz에서 FDR의 의미

- Lorenz에서 현재값 s_t = (x, y, z)(t)는 상태 그 자체다. 그래서 현재값 디코더를 정확히 학습하는 것과 FDR을 정확히 학습하는 것은 같은 상태 정보를 요구한다.
- FDR이 주는 것은 추가 출력 제약이지, 새로운 독립 상태 정보가 아니다.
- 따라서 이번 실험은 학습 목표, 디코더 기하, 마스크 일관성, 상태 복원 정확도의 변화를 본다. FDR의 본질적 필요성은 부분 관측계(문서 D)에서 시험한다.

## 3. 구현 구조와 변경 범위

**새 파일** (모두 신규, 기존 파일 수정 없음)

| 경로 | 내용 |
|---|---|
| `experiments/fdr/model.py` | `FDRModel`: C와 같은 `MultiViewEncoder`(view별 인코더 + 스칼라 관측수 가중 평균), 잠재 k. 디코더 `group_mlp(n, k, hidden, m)`. `forward()` = lag-0 (기존 `impute`, `run`과 호환), `forward_full()` = (B, K, n, m) |
| `experiments/fdr/loss.py` | `multiplicity`, `fdr_masked_loss`, `sample_hide` (Q 규칙), `make_forward_batch(Q, target)` |
| `experiments/fdr/impute.py` | `impute_fdr(mode="past"\|"bidir")` |
| `experiments/fdr/diag.py` | 연속 일관성, 참 상태 near-collision, 탐침, 전체 지연 오차 |
| `experiments/fdr/tests/test_fdr.py` | 테스트 7개 |
| `results/extra/phase3b_fdr_pilot/{pilot_train.py, run_lane.sh, pilot_eval.py, existing_ckpt_diag.py}` | 실행·평가 스크립트 |

`src/`와 `baselines/` 해시는 그대로다 (`md5sum baselines/fprm/*.py src/*.py | md5sum` = a1c7edd0). 학습은 `src.train.run`을 그대로 쓰고, 프로세스 안에서 `build_model`과 `forward_batch`만 대체한다 (기존 E, F와 같은 방식).

## 4. 중복도와 손실의 정확한 정의 [Verified]

창 길이 W = span + K, span = (m−1)τ, anchor t ∈ [span, W−1]. lag k 출력은 u = t − kτ를 예측한다.

**c(u) = #{k : span ≤ u + kτ ≤ W−1}**

명제 (증명):
1. **c(u) = m ⇔ u ∈ [span, K−1].** k = 0이면 u ≥ span, k = m−1이면 u ≤ W−1−span = K−1. 그러므로 **c = m인 위치가 존재 ⇔ K ≥ (m−1)τ + 1.** Phase 3A의 "K ≥ (m−1)τ"는 경계에서 하나가 부족했다. K = (m−1)τ이면 그런 위치가 없다 (테스트로 확인).
2. **모든 u ∈ [0, W−1]에서 c(u) ≥ 1 ⇔ K ≥ τ.** u < span이면 k = ⌈(span−u)/τ⌉ ≤ m−1이 존재하고, u + kτ < span + τ ≤ W−1 ⇔ K ≥ τ. 반대 방향은 u = W−1−(K−1)… 경계 사례로 테스트에서 확인. c(u) = 0인 위치는 손실의 분자와 분모 모두에서 제외한다.
3. **최대 중복도에 도달하는 위치는 필요 없다.** 1/c(u) 가중으로 숨긴 원시 항목마다 총가중이 1이기 때문이다. 그래서 K ≥ (m−1)τ+1을 구현 제한으로 두지 않는다. 현재 설정 (m, τ, K) = (8, 1, 64)는 c = m인 위치가 57/71이다.

**손실 (세그먼트 b):**

    L_b = Σ_{u,i} H[u,i] 1{c(u)>0} (1/c(u)) Σ_{k: anchor 존재} (x_i(u) − Ŷ[u+kτ−span, i, k])² / Σ_{u,i} H[u,i] 1{c(u)>0}

전체 손실 = 분모 > 0인 세그먼트의 평균 (`src.train.masked_loss`와 같은 규약). **분모는 원시 항목 수 기준**이다.

- **목표는 원래 관측된 값에만 한정된다:** H ⊆ M (H = M · Bernoulli). 원래 결측과 평가 정답(M = 0)은 목표가 될 수 없다 (테스트 2의 assert).
- **세그먼트 경계:** 세그먼트 앞쪽 span개 위치는 lag>0의 목표로만 쓰이고, c(u)가 작다. 1/c 가중으로 공정하다. 세그먼트 밖 값은 쓰지 않는다.
- τ > 1인 경우도 테스트한 (m, τ, K) 다섯 조합에서 성립했다: (8,1,64), (4,3,20), (5,2,9), (3,5,4), (8,5,36).

## 5. Q (view-dropout)의 코드 수준 정의 [Verified]

`sample_hide(cfg, M, gen, Q)`:
1. H = M · 1{U < hide_rate} (항목별, hide_rate = 0.2).
2. Q > 0이면 세그먼트마다 확률 Q로 view 1개(확률 1/2) 또는 2개(확률 1/2)를 무작위로 남긴다. **탈락한 view의 원래 관측 항목 전부를 H에 추가**한다 (목표가 됨).
3. 남은 view는 자연 MCAR 마스크와 항목 은닉을 그대로 유지한다. **완전한 view를 만들어 넣지 않는다.**
4. 입력 B = M − H.

모든 조건(C0, C15, FDR0, FDR15)에 같은 규칙과 같은 난수 호출 순서를 쓴다. 그래서 같은 seed면 같은 배치와 같은 H를 받는다.
- Q = 0은 `src.train.forward_batch`와 **손실값이 비트 단위로 같다** (테스트 5b).
- Q > 0은 보관된 Model E의 `forward_batch_viewdrop`과 같다 (테스트 5b).

## 6. 테스트 결과 [Verified]

`PYTHONPATH=.pydeps python -m experiments.fdr.tests.test_fdr` → `results/extra/phase3b_fdr_pilot/test_fdr.log`, **7/7 PASS** (GPU 실행 전).

| 테스트 | 내용 |
|---|---|
| 1 Shape/Alignment | 고유값 램프 시계열로 V[t, i, k] = x_i(t − kτ)를 5개 (m, τ, K)에서 확인. 오라클 출력 손실 = 0, 한 칸 어긋난 출력 > 0. 출력 형상 (B, K, n, m) |
| 2 Mask leakage | H와 원래 결측 위치 값을 ±10³ 수준으로 바꿔도 인코더 출력이 비트 동일 (Q = 0, 0.15). 평가 시 비관측 값을 바꿔도 past/bidir 대치 동일. H ⊆ M |
| 3 Multiplicity | c(u) 벡터 = 전수 계산. 벡터화 손실 = 순수 루프 참조 구현 (오차 < 1e−5). "c = m 존재 ⇔ K ≥ (m−1)τ+1", "모두 덮임 ⇔ K ≥ τ". 상수 오차 e → 손실 e² (총가중 1). H = 0 → 손실 None |
| 4 Past-only causality | t₀ 이후의 값과 마스크를 바꿔도 past 출력 [0, t₀] 동일. bidir는 [0, t₀−h] 동일하고, (t₀−h, t₀]은 실제로 바뀜 (미래 사용 확인). (m, τ) = (8,1), (4,3) |
| 5 Gradient/stability | 모든 인코더·디코더 파라미터에 유한하고 0이 아닌 기울기. 관측률 5%에서도 안정. 전부 결측 입력에서 출력 유한. 목표 0개 → None (NaN 없음) |
| 5b Reference equality | 위 §5 |
| 6 Identity sanity | 완전 관측 사인파 복사 과제에서, 짧게 학습한 FDR의 lag-k 출력이 올바른 정렬에서 오차 < 0.01, 한 lag 어긋난 정렬에서 10배 이상 큼 |

**테스트 설계 수정 (구현 수정 아님):** 테스트 6의 첫 판은 느린 사인파(0.07 rad/step)를 써서, 한 lag(τ = 2) 차이가 값을 거의 바꾸지 않아 실패했다 (right 0.0028 vs wrong 0.0070). 주파수를 0.9, 0.55 rad/step으로 바꿨다. 코드 주석에 기록했다.

**진단 코드 버그 (발견 후 수정):** `diag.py`의 FDR 전체 지연 오차가 처음에는 정답으로 **마스킹된 입력 창**(비관측 = 0)을 썼다. 그래서 lag-0 오차가 0.83으로, 같은 모델의 lag-0 상태 오차 0.11과 모순됐다. 정답을 비마스크 X의 창으로 바꾸고, "lag-0 전체 지연 오차 = 상태 오차"를 assert로 추가했다. 버그 판 출력은 `pilot_eval_v1_BUGGY_fdr_metrics.log`로 보존했다. 기존 체크포인트 진단(`existing_ckpt_diag.*`)은 이 항목을 쓰지 않으므로 영향이 없다.

## 7. 파이프라인 재현 [Verified]

새 경로로 학습한 C0 (dt = 0.05, p = 0.7, seed 0)의 test RMSE **0.154441699385643**이 보관된 `lorenz_dt0.05_C_p0.7_s0`의 기록값과 **소수점 끝자리까지 일치**한다. 데이터, 마스크, 분할, 정규화, 설정, 학습 경로가 기존과 같다는 증거다.
