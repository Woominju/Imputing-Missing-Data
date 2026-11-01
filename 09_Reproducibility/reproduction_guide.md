# 재현 안내

- **환경:** `environment/versions.txt`(Python 3.10.20, torch 2.5.1+cu124, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, GPU L40S, 드라이버 580.196.07)
  - 전체 패키지 목록은 `environment/pip_freeze_reef_gpu.txt`에 있다.
  - 코드는 `PYTHONPATH=.pydeps:.`로 실행한다. `.pydeps`는 포함하지 않았으며 `requirements.txt`로 다시 설치한다.
- **코드:** `code_snapshot/`에 프로젝트의 모든 코드가 있다(`experiments/`, `src/`, `baselines/`, `configs/`, Phase 2.5–4의 결과 디렉터리 안 스크립트, 약 2 MB).
  - 프로젝트 루트에 풀어 쓰면 경로가 그대로 맞는다.
  - Phase 7–9의 `code_md5()`는 고정 파일 목록을 읽으므로, 그 파일들(`train7.py`, `train8.py` 등)이 모두 있어야 한다. 모두 포함돼 있다.

**포함되지 않아 완전 자체 재현이 아닌 것**
- 궤적 자료: 생성기와 seed 규칙으로 다시 만든다.
  - Phase 5: seed base 20261028
  - Phase 6: `gen6.s0_of`, SEED6 = 30261029 + 100000κ + id
  - Phase 7: id 40000+
  - Phase 8: id 50000+, 풀 60000+, k-NN 70000+
  - Phase 9: id 200000+
  - 각 phase의 `gen*.py`로 생성한다. Phase 8과 9의 자료 해시는 각 검증 번들의 `FULL_DATA_LIST`에 있다.
- 학습 checkpoint(약 1.4 GB)와 Phase 8 oracle(905 MB)은 학습으로 다시 얻어야 한다(GPU 필요). 해시는 각 phase의 고정 파일과 FULL_DATA_LIST에 있다.
- 실제 자료(`Data/`)는 라이선스가 기록되지 않아 제외했다. 상세는 `EXCLUDED_FILES_INDEX.md`에 있다.

**서버 없이 재검산할 수 있는 것**
- Phase 8: `07_Independent_Verification/phase8_postrun_verification_bundle_20261031.zip` 안의 `audit_scripts/recompute_from_bundle.py`가 모든 확증 판정을 다시 계산한다.
- Phase 9: `07_.../phase9_verification_bundle_20261031.zip`에 `eval/` 복제본 합 480개와 검증 스크립트가 있다.
- Phase 5–7: `07_.../phase{5,6,7}_*/`의 검증 번들(각 MANIFEST, SHA256SUMS 포함)
- 이 아카이브 자체의 무결성: `sha256sum -c 09_Reproducibility/sha256_manifest.txt`(아카이브 루트에서 실행)
