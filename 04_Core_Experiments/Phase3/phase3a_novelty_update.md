# Phase 3A — Novelty Update (2026-10-27)

이전 평가: `phase25_literature_review.md`, `phase25_architecture_decision.md` §7.

## 1. Lv et al. (2025) 확인 결과

- **서지 (Crossref, Semantic Scholar API로 확인):** Jingchan Lv, Hongcun Mao, Yu Wang, Zhihai Yao. "Reconstruction and Prediction of Chaotic Time Series with Missing Data: Leveraging Dynamical Correlations Between Variables". *Mathematics* 13(1):152, 2025-01-03 온라인. DOI 10.3390/math13010152. Gold OA, CC BY.
- **본문 접근: 실패.** MDPI 페이지와 PDF가 이 서버의 자동 요청에 HTTP 403을 반환했다. 브라우저 위장 다운로드는 봇 차단 우회로 판정되어 실행이 거부됐다. 우회는 시도하지 않았다.
- **초록 수준에서 확인된 내용 (API 요약):**
  - Non-Stationary Transformer와 LightGBM을 결합한다.
  - LightGBM이 **변수 간 결합 관계**를 모형화해 결측값을 복원하고, Transformer는 예측에 쓴다.
  - 소산계와 보존계 혼돈계로 실험한다.
  - 결측률 30–50%에서 예측 오차가 비교적 낮다고 보고한다.
  - 특징 중요도가 계의 알려진 동역학과 맞는다고 주장한다.
  - 참고문헌에 Lorenz 원논문, SAITS, reservoir observer(관측되지 않은 변수 추론), "Predicting chaotic dynamics from incomplete input via reservoir computing" 등이 있다.

| 확인 요청 항목 | 상태 |
|---|---|
| 구체적 결측 설정 (위치, 구간/무작위) | **확인 불가** (본문 필요) |
| 모든 변수에 독립 MCAR을 적용하는가 | **확인 불가** |
| 완전 관측 기준 변수가 필요한가 | **확인 불가.** 초록의 "변수 간 결합으로 복원"은 다른 변수의 관측값을 입력으로 쓰는 회귀를 시사한다. 동시 결측 처리 여부는 모름 |
| Delay embedding을 직접 쓰는가 | **확인 불가.** 초록에는 Takens나 지연 임베딩 언급이 없음 (API 요약 기준) |
| 공유 잠재 상태를 학습하는가 | 초록 기준 **아님으로 보임** (LightGBM 회귀 + Transformer 예측) |
| View 간 일관성이나 미분동형성 검증 | 초록 기준 언급 없음 |
| Lorenz, Coupled Lorenz 포함 | 사용자 정보로는 포함. 서버에서 본문 확인은 못 함 |

**필요한 조치:** 사용자가 브라우저로 PDF를 받아 `third_party/lv2025/`에 넣어 주면, 위 항목을 원문으로 확인해 이 문서를 갱신하겠다.

## 2. 갱신된 차별성 평가

| 선행연구 | 겹치는 부분 | 우리와 다른 점 (확인된 범위) |
|---|---|---|
| FPRM (2026) | 지연 임베딩 + 학습 매핑으로 결측 복원 | 완전 관측 변수 1개 필요. 변수별 GP. Lag FPRM은 동시 결측 구간 한정 |
| Lv et al. (2025) | 혼돈계, 다변량 결측, **변수 간 동역학적 상관을 이용한 복원** | 초록 기준으로 트리 회귀 + Transformer. 공유 잠재와 지연 임베딩 이론은 없어 보임. **결측 설정은 미확인** |
| Bakarji et al. (2023) | 완전 지연벡터를 복원하는 AE | 단일 완전 관측 변수, 결측 없음, 목적은 방정식 발견 |
| Latent CCM (2021) | 모든 차원에 결측, 잠재가 끌개를 매장한다는 논증 | 인과 추론이 목적. 변수별 GRU-ODE 잠재 (융합 없음), 지연 임베딩 사용 안 함 |
| MVAE / MoPoE / MIWAE / Partial VAE / GP-VAE | 부분 관측 다중 모달·변수의 공유 잠재, 관측 데이터 ELBO | 동역학과 지연 임베딩의 이론이 없음. 식별가능성 분석 없음 |
| Knockout (2024) | 마스크 입력 + 무작위 마스킹 학습 | 시계열·잠재 구조 없음. 우리 학습 방식의 정당화로 쓰임 |
| Reservoir observer 계열 (Lv et al. 참고문헌) | 관측되지 않은 변수를 동역학 학습으로 추론 | **추가 확인 필요** (이번에 조사하지 않음) |

**정직한 결론:**
- 구성 요소는 모두 기존 기법이다.
- "혼돈계 다변량 결측 복원에서 변수 간 동역학 관계를 쓴다"는 응용 문제 자체도 Lv et al. 2025가 이미 다룬다 (초록 기준).
- 남는 차별성 후보는 다음 두 가지다. 아직 실험으로 입증되지 않았다.
  1. **완전 관측 변수 없이, 모든 변수의 지연 임베딩을 하나의 공유 잠재로 사상하고, 그 일관성을 식별가능성 조건과 함께 검증하는 것.**
  2. **식별 불가능(Lorenz Γ)을 명시적으로 다루는 평가 체계.**
- 이번 Phase 3A 진단은 현재 C가 1을 **아직 만족하지 못함**을 보였다. 분포 밖 패턴의 불일치와, 분포 안의 출력 동치 잠재 변동(Case A) 때문이다.
- Lv et al.의 본문을 확인하기 전까지 "완전 관측 변수 없는 복원" 자체를 새롭다고 주장하지 않는다.
