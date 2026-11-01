# Phase 9 — 독립 검증 (2026-10-31)

**코드:** `results/audit/phase9/verify9_independent.py`
- 프로젝트 모듈을 import하지 않는다(numpy, pandas, scipy만 쓴다).
- **출력:** `results/audit/phase9/out/`(`verify9_{pilot,main}_checks.csv`, `_log.txt`, `_paired.csv`, `verify9_main_sequence.csv`)

## 검증 범위와 결과

| 항목 | pilot | main |
|---|---|---|
| 셀별 복제본 수(6 / 24) | OK | OK |
| 복제본 × 방법별 합동 대치 RMSE(eval 합에서 재계산) | OK (168) | OK (672, 최대 오차 2e-16) |
| 복제본 × 특징별 ρ_ret | OK (96) | OK (384) |
| 모든 셀의 대응 대비(평균, LCB, UCB) | OK | OK |
| 확증 순서 P-1..P-4(판정, 평균, UCB) | — | OK |
| **L2: 원자료에서 마스크를 재생성하고 결측 위치 수 n 재계산(모든 층)** | OK (61,440) | OK (245,760, 오차 0) |
| **L2: zero와 LOCF SSE 재계산** | OK | OK (상대 오차 ≤ 9.5e-13) |
| 학습 행: 수, valid, 상한, seed 규칙, λ(P9 = 1, 나머지 = 0), tower 목표 > 0, smoke 아님 | OK (144) | OK (576) |
| 임시 파일 0, 실패 표지 0 | OK | OK |
| Phase 9 궤적 600개가 서로, 그리고 P6–P8 모든 궤적과 다름(첫 상태) | OK | OK |

**종합:** pilot 11/11, main 14/14 일치

## 검증 수준의 한계
- **AE와 readout 출력 SSE:** eval9가 쓴 합(L1 수준)만 재계산했다. checkpoint 재추론과 R1 재적합은 하지 않았다(재적합은 학습이다).
- **ρ_ret:** 잠재를 저장하지 않으므로 합 수준에서만 재계산했다. 백색화와 10-NN의 독립 재구현은 하지 않았다. 코드 검토만 했다(`eval9.py`).
- **누출 방지(숨은 y 미사용, teacher no-grad, 원래 결측 위치 한정):** 코드 속성이며 `phase9_test_log.txt` 테스트 16/16으로 확인했다.
- **Jena:** 서술 실험이라 독립 재계산 대상에서 제외했다. 학습 행 112개가 모두 valid이고 smoke가 아님만 확인했다.

## 실행 무결성 기록
- **고정:** `PROTOCOL_FREEZE_p9.txt`(40 해시)와 `PROTOCOL_FREEZE_p9_jena.txt`(추가 파일) 모두 실행 전후에 검증 OK다.
- **고정 후 변경:** 고정된 파일의 변경은 없다. 추가된 파일은 jena9.py 하나다(`FREEZE_CHANGES.md`).
- **고정 전 수정:** smoke 테스트 중 발견한 sched9 종료 조건 버그(마지막 작업 실행 중 종료)를 고정 전에 수정했다.
- **완료 현황:** pilot과 main 모두 중단 없이 완료됐다. 계획 중지와 재실행은 없었다.
- **GPU:** 0–3(PCI_BUS_ID 순서), 드라이버 580.196.07. GPU는 seed와 자료 정의에 영향이 없다(코드).
- **학습 불안정성:** κ=0, p=0.3 P9 3/24 run에서 최적 epoch 후 L_tow가 발산했다. 평가 checkpoint는 발산 전 최적 상태이고, 사전 규칙상 유효하다. `phase9_main_results.md` §2에 있다.
