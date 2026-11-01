점검 시각: 2026-10-28 21:15–21:17 KST. GPU는 사용하지 않았다(점검 시 두 GPU 0 MiB). 새 학습도 하지 않았다.

**1. 무결성 테스트 (원 프로젝트에서 실행)**
- `test_phase6.py` 9/9 PASS
- `test_phase5.py` 8/8 PASS
- `test_external.py` 3/3 PASS
- 이 테스트들은 궤적 npz가 필요하므로 ZIP만으로는 실행할 수 없다(npz는 해시만 포함).

**2. ZIP만으로 재현한 분석 (CPU, `python -I`)**
- 이 번들만 풀어서 `analyze_g1.py`, `analyze6.py`, `s1r/analyze.py`를 실행했다.
- 다음 출력이 원본과 **바이트 단위로 같다**:
  - `g1/analysis_g1.csv`
  - `analysis/g2_contrasts.csv`
  - `analysis/g2_did.csv`
  - `analysis/g2_fixed_sequence.csv`
  - `analysis/g2_replicate_metric.csv`
  - `analysis/g3_criteria.csv`
  - `analysis/g3_accuracy_per_rep.csv`
  - `s1r/s1r_results.csv`

**3. MANIFEST 대조**
- 포함 파일 3,423개의 SHA-256이 ZIP 추출본·원본과 모두 일치한다.
- 제외 파일 4,568개는 표본 23개를 다시 계산했고, 모두 일치한다.
