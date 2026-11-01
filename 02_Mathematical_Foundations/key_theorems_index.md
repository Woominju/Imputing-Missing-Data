# 수학 결과 색인

원문은 `source_documents/`에 있다. 증명, 인증, 가설을 구분한다.

| # | 명제 | 종류 | 가정 | 문서 |
|---|---|---|---|---|
| T1 | Lorenz 대칭군. γ₁₂는 모든 κ에서, γ_j는 κ = 0에서만 흐름과 가환한다. 결합 법칙도 불변이다 | 증명 | 초기분포의 대칭, MCAR | `phase8_mathematical_protocol_v2.md` §1.3 L1, `phase5_mathematical_foundations.md` Lemma S |
| T2 | z만 쓰는 지연 사상은 비단사다(적어도 2:1) | 증명(반례) | Γ 대칭 | `math_review.md` §2(b), `phase6_mathematical_errata.md` E1 |
| T3 | 포획 V ≤ 3080.53. Φ^t는 상으로의 해석적 미분동형이다. P ≠ ν(콤팩트 불변집합은 측도 0) | 증명 | σ ≥ 1, β ≥ 2, κ ≥ 0 | protocol v2 §1.3 L2, L3 |
| T4 | N1-X0: R*_X0 ≥ (1/6)Σ E s²/σ². N1-L_j0(κ = 0) | 증명. 수치는 추정 | γ 불변, MCAR, 고정 정규화 | protocol v2 §3, Phase 8 감사 §A2, D3 |
| T5 | B1: R(h) − R* ≥ R(h) − R(f). B3: 잠재와 raw의 차이는 Bayes 손실이 아니다 | 증명 | f가 test와 독립인 고정 함수 | protocol v2 §2.2 |
| T6 | P8.1 κ = 0 인수분해, P8.3 Sham 교환가능성, P8.4 인증 | 증명 | 하위계 독립, donor 독립 | protocol v2 §5.1 |
| T7 | 반례 C1–C5 | 증명 | — | protocol v2 §3 |
| T8 | P7: 점 일관성은 정보를 붕괴시킨다 | 증명 | — | `aim_mathematical_foundation.md:186–190` |
| T9 | A2.2: 무조건 cross-mask 정렬과 구별성은 양립하지 않는다 | 증명 | 비식별 마스크가 존재 | `phase9_mathematical_foundations.md` §A2 |
| T10 | A3.1: 정확한 등식이면 T는 C¹ 임베딩이다. A3.2: 근사 단사성 | 증명(조건부) | Φ가 임베딩(미검증), D가 Lipschitz | 같음 §A3 |
| T11 | A4.1, A4.2: holdout 손실의 Fisher 일관성. 전체 위험 ≤ 관측 초과위험 / π_min | 증명 | MCAR, 양성 조건 | 같음 §A4 |
| T12 | A4.3: tower 자기 증류의 유일 고정점. 참값 항목이 없으면 붕괴 | 증명(무한 용량) | 위와 같음 | 같음 §A4 |
| C1 | Δ*_L0 > 0(Phase 6 G1, LCB 0.0322) | **인증 → Empirical로 강등** | bootstrap 근사 | `phase6_g1_results.md`, `phase7_mathematical_corrections.md` §A5 |
| C2 | Phase 8 I-1: 모델 한계 ≥ 0.041 MSE | 인증(통계 LCB) | 풀 조건부 | Phase 8 감사 §D |
| C3 | Phase 8 IIa-2: φ_Bfull(W)는 W_j의 함수가 아니다(L_j0) | 인증 | N1-own | Phase 8 감사 §E |
| W1 | MCAR ⇒ 가우스 PoE 정확 | **철회** | — | `phase25_mathematical_audit.md` |
| W2 | ELBO 최적 ⇒ 참 사후분포 | **틀림**(조건부 ELBO 아님) | — | `phase25_objective_redesign.md` |
