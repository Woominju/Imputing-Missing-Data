# Phase 3D-1 — Document B: Evaluation Integrity (2026-10-27)

원칙:
- 원본 코드, CSV, 보고서는 수정하지 않았다.
- 새 산출물은 `results/extra/phase3d1_audit/`에만 있다.

## 1. 기존 CSV 검증 [Verified]

- **파일:** Phase 3C의 모든 CSV가 존재하고 구조가 정상이다(이전 단계의 `RECOVERY_MANIFEST.md`).
- **재계산 일치** (`repro_checks_errordecomp.csv`, `probe_determinism_check.csv`):
  - lag-0 출력은 CPU 재계산과 GPU 저장값의 차이가 3.2e-6 이하다.
  - test 대치 RMSE는 `runs.csv`와 5e-8 이하로 일치한다.
  - 같은 상태 dR_q50은 2e-9 이하로 일치한다.
  - ridge와 kNN probe는 16개 행(xz 6개 모델 × 2 probe + x-only 2개 × 2) 모두 test RMSE와 선택된 하이퍼파라미터가 **비트 단위로 일치**한다.
  - `eval_dynamics.csv`의 미래 y 값 16개도 원래 로직으로 재계산하면 정확히 일치한다.
- **MLP probe는 다시 돌리지 않았다.** 이유는 §6에 있다.

## 2. z-removal 버그 집계 문제의 해결

- **문제:** `analyze.py`는 원본 `eval_probe.csv`를 그대로 읽는다. 그래서 무효인 `*_zremoved_refit` 18행이 집계(`analysis.log`)에 들어간다.
  - 이 18행은 실제로는 x를 제거한 결과다. `*_xremoved_refit`과 비트 단위로 같다.
- **해결:** 원본 `analyze.py`는 그대로 두고, 새 집계 스크립트 `results/extra/phase3d1_audit/safe_aggregate.py`를 만들었다. 이 스크립트는 다음을 assert로 검사하고, 하나라도 어긋나면 실패한다.
  1. 원본 `eval_probe.csv`의 무효 행이 정확히 18개이고, x-removed 행과 동일하다(버그의 지문). 지문이 바뀌면 다시 감사해야 하므로 멈춘다.
  2. 무효 행을 제외한 결과가 `recovery/eval_probe_valid_rows.csv`와 완전히 같다.
  3. z 제거 결과는 `eval_probe_zremoved.csv`(36행, mode 2종)에서만 가져온다.
  4. 출력 `probe_summary_safe.csv`에 `_zremoved_refit` 문자열이 없다.
- 실행 결과는 rc=0이다(`safe_aggregate.log`).
- **앞으로 probe 결론은 이 스크립트의 출력만 사용한다.** `analysis.log`의 z-removed 줄은 무효다.

## 3. 미래 y target의 시간 경계 감사

**코드** (`phase3c_hidden_state/evaluate.py` 86–90행):
- `yD = r_[yn[D:], nan*D]`로 전체 배열을 먼저 이동한 뒤, 각 split의 t를 고르고 `isfinite`만 확인한다.
- 그래서 **t ∈ [a, b) 이면서 t + D ≥ b인 행이 남는다** [Verified, 코드와 재계산].

**누수 행 수** (`eval_dynamics_boundary_fixed.csv`):

| target | train 행 중 target이 val에 있는 행 | val 행 중 target이 test에 있는 행 | test |
|---|---|---|---|
| y(t+1) | 1 / 13,993 | 1 / 3,000 | 0 (배열 끝이라 NaN으로 제외) |
| y(t+5) | 5 / 13,993 | 5 / 3,000 | 0 |

**영향:**
- val 행 D개는 test 구간의 첫 D개 y를 target으로 썼다. 이것은 kNN의 k 선택에만 쓰였다.
- train 행 D개는 val의 y를 학습 target으로 썼다.
- test 예측 자체에는 누수가 없다.

**수정 평가** (t + D < b, `dynamics_boundary.py`):
- 16개 값 모두 |수정 − 원본| ≤ **2.3e-4**다.
- 선택된 k는 16개 모두 같다.
- C/FDR 순서, 즉 "1 step에서는 FDR이 낫고 5 step에서는 섞여 있음"도 그대로다.

**결론 [Verified]:** 경계 누수는 실제로 있었지만, 크기는 D행 수준이고 수치 영향은 1e-4 단위이며 결론은 바뀌지 않는다.
- 기존 `eval_dynamics.csv`는 보존했다.
- 수정값은 `eval_dynamics_boundary_fixed.csv`의 `fixed` 열에 있다.

## 4. 그 밖의 누수와 재현성 위험 점검

| 항목 | 상태 |
|---|---|
| y가 AE 데이터, 학습, 모델 선택에 들어감 | 없음 [Verified, Phase 3C 누수 테스트 6개] |
| 정규화와 probe 표준화 | train 통계만 사용 [Verified, 코드] |
| 입력 창이 split 경계를 넘음 (val/test 첫 7개 anchor가 이전 split의 x, z를 입력으로 사용) | **레이블 누수는 아니다**. x, z는 관측 입력이고 y가 아니다. AE 학습 segment는 train 안에만 있다(`max_start = b − W`). 기록만 해둔다 |
| y probe 행의 경계 | 현재 시점 y(t)만 쓰므로 split 내부다 [Verified] |
| 같은 상태 일관성의 kNN probe | 자연 마스크의 train 잠재에만 적합했다 [Verified] |
| test 구간 재사용 | Phase 3C와 3D-1에서 같은 test 구간으로 여러 분석을 했다. **test로 선택하거나 튜닝한 것은 없다.** 다만 비교 횟수가 많으므로 개별 CI는 다중 비교 보정 없이 읽어야 한다 |
| seed | seed 3개는 같은 Lorenz 궤적을 쓴다. 바뀌는 것은 마스크, 초기화, 배치뿐이다. 독립 궤적 반복이 아니다 |
| 사후 탐색 지표 | `mean over lags 0..3`은 이번에 추가했다(탐색용). `mean over lags 0..7`은 Phase 3B `impute_fdr` bidir의 기본 정의다 |
| 랜덤 인코더 대조군 | current와 full의 값이 같다. 같은 시드에서 encoder가 먼저 초기화되고, probe는 encoder 출력만 쓰기 때문이다. 버그가 아니다 [Verified, Phase 3C] |

## 5. 이번에 새로 발견한 해석상 문제 (측정은 정확하지만 해석이 틀렸던 경우)

- Phase 3C의 일관성 지표 d_R(디코더 lag-0 출력 차이)과 `exz`(관측 변수 복원 오차)는 **관측 좌표와 결측 좌표를 섞은 지표**였다.
- 그런데 두 모델 모두 관측 좌표의 lag-0 출력은 학습 손실에 들어가지 않는다(Document A §0).
- 그래서 이 지표들은 "결측값 예측의 일관성"을 측정하지 못한다. 측정 자체는 정확하다. 결론의 범위가 넓었다.
- **수정된 해석:** 결측 좌표에서는 C가 정확도와 일관성 모두 우수하다.
- 수정 대상인 기존 보고서 문장은 다음과 같다. 원문은 수정하지 않았다.
  - `phase3c_state_consistency.md` §1 표와 §5 (a)
  - `phase3c_research_decision.md` §2 항목 3
  - "AIM의 shared latent를 지지하는 첫 번째 정량적 증거"라는 표현도 결측 좌표 기준으로는 성립하지 않는다.

## 6. 기존 결과와 수정 결과 비교 요약

| 기존 결론 (Phase 3C) | 3D-1 판정 |
|---|---|
| 대치 RMSE는 C가 3 seed 모두 우수 | 유지. 결측 좌표만 따로 보면 같은 결과다 |
| 비선형 y-probe는 FDR 우수 | 수치는 유지된다(kNN, ridge 재현 일치). MLP probe는 초기화 seed 1개의 결과라 probe 자체의 분산을 모른다 → 재구성이 필요하다(아래) |
| 선형 y-probe는 C 우수 | 유지. 다만 두 잠재 모두 0.46–0.53으로 나빠서 정보량 비교로 쓰기 어렵다(Document C) |
| 같은 상태 출력 일관성은 FDR 우수 | **관측 좌표에 한정된다.** 결측 좌표에서는 C가 우수하다 |
| 미래 y, 1 step에서는 FDR 우수 | 유지. 경계 누수 수정의 영향은 ≤ 2.3e-4 |
| z 제거 시 y 오차 약 2배 | 유지. 올바른 CSV에서 나온 값이다 |

**재구성이 필요한 항목** (이번에는 실행하지 않음; 모두 CPU로 가능):
- (a) **MLP probe 초기화 seed 반복.** probe seed를 5개 이상 돌려 FDR − C 차이가 probe 분산보다 큰지 확인한다. 기존 MLP 예측값은 저장되지 않았다.
- (b) probe의 test 예측값 저장. 이것이 있으면 paired bootstrap을 probe 단위로 다시 할 수 있다.
- (c) Phase 3C 블록 CI의 재확인. 기존 블록 RMSE는 `eval_blocks.csv`에 있다.
