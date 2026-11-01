# Official AIM Alignment

> We aim to develop a dynamics-informed imputation framework that reconstructs missing values in multivariate time series without requiring any fully observed variable. It does so by learning a shared latent state to which the delay embeddings of all variables are mapped consistently, as implied by Takens' embedding theorem.

상태 정의: Established(수학+코드/자료로 확립) · Partially Supported(일부 확증/서술) · Unresolved(증거 불충분 또는 반대) · Not Evaluated(공정 평가 미수행). 퍼센트·점수는 쓰지 않는다.


| 구성 요소 | 상태 | 근거 |
|---|---|---|
| 완전 관측 변수 없이 대치 | **Established** | A4.1(수학), 모든 모델이 참조 없이 학습(코드). 대치 RMSE가 LOCF와 zero보다 크게 낮음 |
| 다중 view 지연 매장 통합 | **Partially Supported** | 상대 view 정보 존재(인증)와 활용(IIa 확증, 작음). Jena에서 다중 view가 변수별 AE보다 나음(서술). 결합 인코더 우위는 없음 |
| 일관된 공유 잠재 상태 | **Unresolved** | G3·GSR 기준 미달. P9 개입 효과 미미. 비식별 마스크의 원리적 한계(A2) |
| Takens형 수학적 대응·식별성 | **Unresolved** | 조건부 정리(A3.1)만 있음. 학습 모델이 가정을 만족한다는 증거 없음 |
| 시뮬레이션과 실제 자료 검증 | **Partially Supported** | 결합 Lorenz는 사전등록 확증 설계. 실제 자료는 서술(Phase 1 FPRM 자료 5종, Phase 5 Beijing/Jena, Phase 9 Jena) |
| FPRM 및 기존 방법과의 공정 비교 | **Not Evaluated**(공정 비교 기준) | FPRM은 참조 변수를 요구함. 조건 B에서는 Adapted FPRM만 실행됨(Phase 1). Lag-FPRM 미구현, 포트 결함 미해결 |
