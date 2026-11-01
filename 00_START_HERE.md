# 00 START HERE

**먼저 `index.html`을 여세요.** 더블클릭하면 브라우저에서 열리며, 인터넷이나 서버가 필요 없습니다. 연구 전체의 개요, Phase 1–9 연표, 수학 결과, 실험 비교, AIM 진척, 향후 연구를 한 화면에서 볼 수 있습니다.

## 한 문단 요약
- **연구 질문:** 완전 관측 변수 없이, 여러 불완전 지연 view를 하나의 공유 잠재 상태로 통합해 결측을 대치할 수 있는가?(결합 Lorenz, 실제 자료)
- **수학적으로 확립된 것**
  - 참조 없는 학습의 근거(holdout 손실의 Fisher 일관성)
  - 무조건 마스크 간 정렬의 불가능성(Lorenz 대칭)
  - 조건부 임베딩 정리
  - 대칭 기반 정보 하한
- **실험으로 확증된 것**
  - 상대 view의 숨은 상태 정보는 존재하고 일부 활용된다(작음).
  - 감독 범위가 잠재 내용을 결정한다.
  - 관측 정보 대비 모델 한계가 있다(≥ 0.041 MSE).
- **가장 중요한 부정적 결과**
  - 결합 공유 잠재(B-full)가 하위계별 병치(Fact8)보다 낫지 않다(p=0.7 대치에서는 병치가 낫고, 독립 자료로 재현됨).
  - Phase 9의 tower 자기 증류(P9-MVAE)도 개선하지 못했다.
- **AIM 진척:** 공유 잠재 상태와 수학적 대응은 Unresolved, FPRM 공정 비교는 Not Evaluated다.

## 읽는 순서
1. `index.html`: 대시보드
2. `research_key_findings.md`: 10분 요약
3. `01_Research_Synthesis/phase1_to_phase9_research_synthesis.md`: 통합 보고서. 연표와 AIM 정합성 문서도 같은 폴더에 있다.
4. `research_experiment_master_table.csv`: 실험 40건, 15개 열
5. `02_Mathematical_Foundations/`: 정리 색인, 미해결 문제, 원문 수학 문서
6. `04_Core_Experiments/Phase*/`: 각 Phase의 보고서, 프로토콜, 분석 CSV
7. `07_Independent_Verification/`: Phase 5–9 검증 번들과 독립 재계산 스크립트
8. `08_Future_Research/research_continuation_plan.md`
9. `09_Reproducibility/`: 환경, 전체 코드 스냅샷, 재현 안내, 파일 목록과 SHA-256
10. `EXCLUDED_FILES_INDEX.md`: 포함하지 않은 자료와 확보 방법

## 무결성
아카이브 루트에서 `sha256sum -c 09_Reproducibility/sha256_manifest.txt`를 실행한다.
