# 핵심 모델 코드 (사본; 실행용 전체 코드는 `09_Reproducibility/code_snapshot/`)

| 모델 | 정의 | 파일 |
|---|---|---|
| FDR (Full Delay Reconstruction) | view별 인코더 → 공유 잠재 → view별 **전체 지연 벡터** 디코더. 숨긴 관측 항목 H에서만 1/c(u) 중복도 가중 MSE를 쓴다 | `FDR/model.py`, `FDR/loss.py`(fdr_masked_loss), `FDR/delay_embedding.py` |
| ArmModel (B-self, B-full) | FDR + 마스크 인식 학습 fusion. B-full은 4 view 입력과 4 view 손실을 쓴다. B-self_j는 자기 view만 손실에 쓴다(Phase 7) | `Bfull_Bself_ArmModel/core.py`(phase5), `core7.py`, `train7.py` |
| Fact8 / Fact16 | 하위계별 ArmModel 2개(잠재 8 또는 16)를 연결한 병치 | `Fact8_Fact16/core8.py`(arm_spec), `feats8.py`(FEATS), `train8.py` |
| P9-MVAE | B-full + L_tow(teacher = 같은 망, 입력 O, no-grad, 원래 결측 위치에서만 목표, λ = 1) | `P9_MVAE/core9.py`(tower_loss), `train9.py`(p9_loss), `eval9.py` |

- 모두 **완전 관측 참조 없이** 학습한다. 조기 종료도 val의 숨긴 관측 항목으로만 한다.
- 숨은 상태 y는 평가(readout, 검색)에서만 쓴다.
