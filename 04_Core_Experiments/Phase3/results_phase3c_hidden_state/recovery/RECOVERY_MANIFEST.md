# Phase 3C — RECOVERY_MANIFEST (2026-10-27 15:30 KST)

## 결론

- 서버에서 **누락되거나 손상된 Phase 3C CSV는 없다** (Case A). 따라서 재학습도, 평가 재계산도 하지 않았다.
- 이름에 `recovered`가 붙은 파일은 없다. 실제 복구가 일어나지 않았기 때문이다.
- ZIP의 모든 원본 파일은 서버 원본을 바이트 그대로 복사했다. 아래 SHA256은 서버 원본의 해시다.
- 새로 만든 파일은 두 개뿐이다.
  - `eval_probe_valid_rows.csv`: 원본 `eval_probe.csv`에서 버그 행 18개만 제외한 파생 파일. 값은 바꾸지 않았다.
  - 이 매니페스트.

## z-removal 버그

- 첫 `evaluate.py`는 "z 제거" 입력을 만들 때 x 열을 지웠다 (`Mzr[:, 0] = 0`).
  - 그래서 `eval_probe.csv`의 `latent_{C,FDR}-xz_zremoved_refit` 18행(2 모델 × 3 seed × 3 probe)은 `*_xremoved_refit` 행과 비트 단위로 같다. 이번 감사에서 확인했다.
  - 이 행들은 **무효**다.
- 수정 코드:
  - `evaluate.py` 79행이 현재 `Mzr[:, 1] = 0`로 고쳐져 있다. 이 수정은 CSV 생성(12:41) 이후인 12:42에 이뤄졌다.
  - 따라서 원본 `eval_probe.csv`는 수정 전 코드로 만들어졌다.
- 올바른 z 제거 결과는 `eval_probe_zremoved.csv`다.
  - 행 수: 2 모델 × 3 seed × 3 probe × 2 mode = 36행.
  - mode는 `refit_on_z_removed`와 `natural_probe_applied_to_z_removed` 두 가지다.
  - 생성 코드는 `multiview_fix.py`다. 이 코드는 `assert`로 x 마스크가 그대로이고 z 마스크가 0임을 확인한다.
- 보고서 `phase3c_experiment_results.md` §5의 z 제거 수치는 `eval_probe_zremoved.csv`와 일치한다. 즉 보고서는 버그 행을 인용하지 않는다.
  - 재적합 MLP: C 0.106/0.114/0.125, FDR 0.085/0.090/0.109.
  - 전이: C 0.172/0.192/0.184, FDR 0.163/0.153/0.159.
- `analysis.log`의 z-removed 줄(22–37행 블록과 81–86행 블록의 `zremoved`)도 같은 버그 값이다.

## 학습 8회 대조 (runs.csv ↔ 체크포인트 ↔ 학습 로그)

8개 체크포인트 모두 CPU에서 로드된다. 각 run은 다음이 서로 일치한다.

- 파라미터 수가 `runs.csv`의 `n_params`와 같다.
- 학습 로그 행 수가 `epochs_run`과 같다.
- 로그의 val 최소 epoch이 `best_epoch`와 같다.

모든 run은 다음 조건을 만족한다.

- 종료 사유가 `patience`다.
- `hit_cap = False`다.
- `nonfinite_steps = 0`이다.
- `code_md5`가 f85b14f9다.

`eval_imputation.csv`의 `recorded`는 `runs.csv`의 `test_rmse`와 같다.

| run | 체크포인트 SHA256 (앞 16자) |
|---|---|
| p3c_C-x_dt0.05_p0.3_s0.pt | 96e40799afd83c9a |
| p3c_C-xz_dt0.05_p0.3_s0.pt | 394a92df0286c34d |
| p3c_C-xz_dt0.05_p0.3_s1.pt | a19dffd6d567f12a |
| p3c_C-xz_dt0.05_p0.3_s2.pt | a0ac977e93fcad80 |
| p3c_FDR-x_dt0.05_p0.3_s0.pt | d7544ec72cfd1ffa |
| p3c_FDR-xz_dt0.05_p0.3_s0.pt | 01b538c2831594f0 |
| p3c_FDR-xz_dt0.05_p0.3_s1.pt | b0a786c8319b1d88 |
| p3c_FDR-xz_dt0.05_p0.3_s2.pt | 4ba8886cd8a177df |

체크포인트와 `*_xhat_past.npy`는 ZIP에 넣지 않았다. 서버 원본 경로는 다음과 같다.

- `/home/minju/faicpaston/Imputing_Missing_Data/results/extra/phase3c_hidden_state/checkpoints/`
- `/home/minju/faicpaston/Imputing_Missing_Data/results/extra/phase3c_hidden_state/`

## 의도된 NaN (손상 아님)

| 파일 | 열 | NaN 행 수 | 의도된 이유 |
|---|---|---|---|
| `eval_probe.csv` | `coverage` | 108 | FD 기준선 행에만 정의된다 |
| `eval_probe.csv` | `val_rmse` | 6 | 평균 예측기와 FD 해석식에는 검증 선택이 없다 |
| `eval_imputation.csv` | `rmse_z` | 2 | x-only 모델에는 z가 없다 |
| `runs.csv` | `window`, `direction` | 각 8 | cfg에 값이 없다(None) |
| `observability_local.csv` | `cond`, `amp_y` | 각 2087 | 계수 결손(rank < 3) 패턴에서는 정의되지 않는다. 주로 lag-0만 있는 패턴이다 |

그 밖의 CSV는 다음을 확인했다.

- 모든 행의 필드 수가 헤더와 같다.
- 중복 행이 없다.
- 파일 끝이 줄바꿈으로 끝난다(잘린 마지막 행 없음).

## 파일 목록

| ZIP 경로 | 서버 원본 | 행 수 | SHA256 | 상태 |
|---|---|---|---|---|
| `csv/eval_probe_zremoved.csv` | `results/extra/phase3c_hidden_state/eval_probe_zremoved.csv` | 36 | `f5caa6a959d09aed135b0cd87fc2cc314690bc995457512d49e0ff5bbe8b6225` | VALID (original, unchanged) |
| `csv/eval_imputation.csv` | `results/extra/phase3c_hidden_state/eval_imputation.csv` | 8 | `fbd215854a475059e725816aa5297a153144f92982d80a1e7ec2b3367a9ff3c0` | VALID (original, unchanged) |
| `csv/eval_consistency.csv` | `results/extra/phase3c_hidden_state/eval_consistency.csv` | 18 | `2928263d98bfc7369143ad05325f91dcc93069dbd61536b75aca758bb50df8c6` | VALID (original, unchanged) |
| `csv/eval_dynamics.csv` | `results/extra/phase3c_hidden_state/eval_dynamics.csv` | 24 | `b0337a82aa539519577542dfdf033110da84d7f21a1bb4717686ccf22d213d19` | VALID (original, unchanged) |
| `csv/eval_blocks.csv` | `results/extra/phase3c_hidden_state/eval_blocks.csv` | 105 | `967205a54a95e0fc1ce164bd76a0eebe56c2867c975086c401f1c535e441c593` | VALID (original, unchanged) |
| `csv/eval_acf.csv` | `results/extra/phase3c_hidden_state/eval_acf.csv` | 3 | `e61c72206c18534e385910bb85e87d01f350c29e3365d7a8ff43285b2fc40ede` | VALID (original, unchanged) |
| `csv/runs.csv` | `results/extra/phase3c_hidden_state/runs.csv` | 8 | `e3c65b509fd646ef85b59b6a18ecc74f9a156ab6a1eb1affad5b5673a439d737` | VALID (original, unchanged) |
| `csv/observability_local.csv` | `results/extra/phase3c_hidden_state/observability_local.csv` | 7976 | `e2a9320638290ea41e20d0bfd2b4ebf02adf45226ab9dc0af1a3d9c0c536a5a0` | VALID (original, unchanged) |
| `csv/observability_global.csv` | `results/extra/phase3c_hidden_state/observability_global.csv` | 4200 | `11ca5b7fcf569fb20638fbdddf244d72c28edb892d7b80b784597e64437aad0f` | VALID (original, unchanged) |
| `csv/fd_baseline.csv` | `results/extra/phase3c_hidden_state/fd_baseline.csv` | 12 | `a1e9d0e9f4a51bc62f497f7dd663a6ddaecd928dafdd0c679af6241ae66e0b5b` | VALID (original, unchanged) |
| `csv/eval_probe_valid_rows.csv` | `results/extra/phase3c_hidden_state/recovery/eval_probe_valid_rows.csv` | 93 | `cdd33a3ded8d99a4181611461df3abc2bfcbe8d51e2ff4ca68b775a932820cd0` | VALID (derived: eval_probe.csv minus 18 *_zremoved_refit bug rows) |
| `training_logs/p3c_C-x_dt0.05_p0.3_s0.csv` | `results/extra/phase3c_hidden_state/logs/p3c_C-x_dt0.05_p0.3_s0.csv` | 136 | `0c59e3a482bfa8ba608188ce9c1a53f430774cf6be527fe1f6a5d095f4ab4f9b` | VALID (original, unchanged) |
| `training_logs/p3c_C-xz_dt0.05_p0.3_s0.csv` | `results/extra/phase3c_hidden_state/logs/p3c_C-xz_dt0.05_p0.3_s0.csv` | 194 | `704fa8fbf7adbb4f564015aea4714d3da1d25fe1ba9ae4680d4ef8e83ea06a4e` | VALID (original, unchanged) |
| `training_logs/p3c_C-xz_dt0.05_p0.3_s1.csv` | `results/extra/phase3c_hidden_state/logs/p3c_C-xz_dt0.05_p0.3_s1.csv` | 98 | `d8100862604038bc07b57fc14c8d6a3f2d809e807ae2ef0297acb473286ba0f7` | VALID (original, unchanged) |
| `training_logs/p3c_C-xz_dt0.05_p0.3_s2.csv` | `results/extra/phase3c_hidden_state/logs/p3c_C-xz_dt0.05_p0.3_s2.csv` | 102 | `3f7e03c3c3f5348d8784630b3fd8a08f0f8efd2e737e20d7798a237b5b293344` | VALID (original, unchanged) |
| `training_logs/p3c_FDR-x_dt0.05_p0.3_s0.csv` | `results/extra/phase3c_hidden_state/logs/p3c_FDR-x_dt0.05_p0.3_s0.csv` | 146 | `c32a81e200244371acee99dee740c4f39aa172ffa64ba1947b857834714c3244` | VALID (original, unchanged) |
| `training_logs/p3c_FDR-xz_dt0.05_p0.3_s0.csv` | `results/extra/phase3c_hidden_state/logs/p3c_FDR-xz_dt0.05_p0.3_s0.csv` | 136 | `014b79e51220e20a4981c5931dfa14b8d6c6b1d65b7277a36fe16f4a5d0fb042` | VALID (original, unchanged) |
| `training_logs/p3c_FDR-xz_dt0.05_p0.3_s1.csv` | `results/extra/phase3c_hidden_state/logs/p3c_FDR-xz_dt0.05_p0.3_s1.csv` | 130 | `c1bdc8e975e92ba9db88a30b8d3cff9ef8a866cc3be6aef8e365572ae08a00c4` | VALID (original, unchanged) |
| `training_logs/p3c_FDR-xz_dt0.05_p0.3_s2.csv` | `results/extra/phase3c_hidden_state/logs/p3c_FDR-xz_dt0.05_p0.3_s2.csv` | 91 | `65281e02e9d61816322ef6596cad003b8bdef07bbf9b43900c4e6af83bbcceea` | VALID (original, unchanged) |
| `invalid_or_superseded/eval_probe.csv` | `results/extra/phase3c_hidden_state/eval_probe.csv` | 111 | `f0f81f04daf59e33e880d0e9f9e7321171213d9ad58194edbf217101fe4a0f40` | MIXED (original, unchanged): 93 valid rows + 18 invalid *_zremoved_refit rows |
| `invalid_or_superseded/analysis.log` | `results/extra/phase3c_hidden_state/analysis.log` |  | `1ee39acb5a448683e884559a1397749b77e727e943c48f9a6bcc15daab47920e` | MIXED (original, unchanged): z-removed lines show the buggy x-removed values |
| `notes/phase3c_observability_audit.md` | `results/notes/phase3c_observability_audit.md` |  | `de7ed0609c4e699c9c5cd8bb831089f51fe6bf149825aee1c98551c708a45c29` | REPORT (original, unchanged) |
| `notes/phase3c_experiment_results.md` | `results/notes/phase3c_experiment_results.md` |  | `9bb77ae9d63d48b817258735f927902edf36a82504a2b7767e2f19e5683e30ce` | REPORT (original, unchanged) |
| `notes/phase3c_state_consistency.md` | `results/notes/phase3c_state_consistency.md` |  | `f454805d8eee8d605c81d3cbea674c34bc8f5f094156c111f2ee5350a43a091b` | REPORT (original, unchanged) |
| `notes/phase3c_research_decision.md` | `results/notes/phase3c_research_decision.md` |  | `bd4e58cd1eaf9a2981981afcf9c1a6d06b4c52b3e2a3c08778301bb00615ac7f` | REPORT (original, unchanged) |
| `notes/phase3c_partial_observation_proposal.md` | `results/notes/phase3c_partial_observation_proposal.md` |  | `1a0a1e56efc8bfacae593186f8171fddd01655582ef99f7e1ff662a29313a486` | REPORT (original, unchanged) |
| `run_logs/tests.log` | `results/extra/phase3c_hidden_state/tests.log` |  | `723a497a236ccc56b458c673c7a3d23c42f60b412700e97c667089ecb767e013` | LOG (original, unchanged) |
| `run_logs/tests_leakage_rerun.log` | `results/extra/phase3c_hidden_state/tests_leakage_rerun.log` |  | `5c014fb9c89cadca314451d4d666b506c1829a77602f4ff41279af8c5b6fefc8` | LOG (original, unchanged) |
| `run_logs/lane_gpu0.log` | `results/extra/phase3c_hidden_state/lane_gpu0.log` |  | `621ed929b1b15e2531f994e49bdd965343ae32a0d5efb8b513e03d9417439e04` | LOG (original, unchanged) |
| `run_logs/lane_gpu1.log` | `results/extra/phase3c_hidden_state/lane_gpu1.log` |  | `c4ab860f2c817afa0f55115222397156076d6dc31bcc863d1101ee2229fe204b` | LOG (original, unchanged) |
| `run_logs/evaluate.log` | `results/extra/phase3c_hidden_state/evaluate.log` |  | `2bba52f5b79afc0a7746a94478d59c4be6a2ce18fdcb9f934603c4d0278d8963` | LOG (original, unchanged) |
| `run_logs/multiview_fix.log` | `results/extra/phase3c_hidden_state/multiview_fix.log` |  | `2160dd6cd1aaac6add4437ede9459b6d8bbd42c1896f26e5caf308d6251d0905` | LOG (original, unchanged) |
| `run_logs/observability_audit.log` | `results/extra/phase3c_hidden_state/observability_audit.log` |  | `c976c106a05b78b2b3e946079430fac57f9120b7007d8c3641407753b1138763` | LOG (original, unchanged) |
| `run_logs/test_artifacts/leaktest.csv` | `results/extra/phase3c_hidden_state/test_artifacts/leaktest.csv` | 2 | `5b4c24af9c4698100c1ea21784ea4c375ae4cb428cb18063200eefb7b6cdc819` | TEST ARTIFACT (not a result) |
| `run_logs/test_artifacts/logs/leaktest.csv` | `results/extra/phase3c_hidden_state/test_artifacts/logs/leaktest.csv` | 2 | `b14a5617a364eb2baf929186698237fe990f2c5bb22409e5219e59cb14d88c80` | TEST ARTIFACT (not a result) |
| `code/train.py` | `results/extra/phase3c_hidden_state/train.py` |  | `7d80ccb422320d027d1d33ee23a80167c93ece6a69ecc098c460a06062652f24` | CODE (original, unchanged) |
| `code/run_lane.sh` | `results/extra/phase3c_hidden_state/run_lane.sh` |  | `91f0f13d872a9b8fdd98b3f41f8e143337b384a30764839aba7c63a1ef523391` | CODE (original, unchanged) |
| `code/evaluate.py` | `results/extra/phase3c_hidden_state/evaluate.py` |  | `1860429a1528a67b37c600eacf7eb2ad169551762cbd8998ab36e4982fbac625` | CODE (original, unchanged) |
| `code/multiview_fix.py` | `results/extra/phase3c_hidden_state/multiview_fix.py` |  | `4502b32c98d1e7a7b14c6e0affc1a17e9fca8cd6e9af2097a11b9c1e75de0f48` | CODE (original, unchanged) |
| `code/analyze.py` | `results/extra/phase3c_hidden_state/analyze.py` |  | `1308a473e259d68755a38acec08f1f991e0ccf114afe2ce0165bdc1f97670976` | CODE (original, unchanged) |
| `code/observability_audit.py` | `results/extra/phase3c_hidden_state/observability_audit.py` |  | `b26575ea4c9465eb0a8d7daa604ea3199faca7a464cf43f509966f1052b382ec` | CODE (original, unchanged) |
| `code/experiments_fdr/xz_data.py` | `experiments/fdr/xz_data.py` |  | `6c7edb9e8a438602a2d4d29342d7fd6fecb81f8c40109f4bf7d5dc2113c094af` | CODE (original, unchanged) |
| `code/experiments_fdr/probes.py` | `experiments/fdr/probes.py` |  | `ee6cef2c4ee1a4ee14f59cd875fe88bfd6a08acbdaf1853ea547fb09b7d51ae2` | CODE (original, unchanged) |
| `code/experiments_fdr/features.py` | `experiments/fdr/features.py` |  | `aea34aa0dd1a8b53531cdeddd4cdfea3c2909bed9771bbf88652007d6970e0f7` | CODE (original, unchanged) |
| `code/experiments_fdr/model.py` | `experiments/fdr/model.py` |  | `8f587d386bd916d82fc7abdaf8b936a21ac303d2b4b3f41fba9b82a7d9c273cf` | CODE (original, unchanged) |
| `code/experiments_fdr/loss.py` | `experiments/fdr/loss.py` |  | `37cac2543bde34e402785a11cd5d17f2b16cef508cf4bdbe89775bcc18e3d2f9` | CODE (original, unchanged) |
| `code/experiments_fdr/impute.py` | `experiments/fdr/impute.py` |  | `541f2e465829f8306835f0e4ea02b4f61a5c82b7410f772b5e243784d1a8a553` | CODE (original, unchanged) |
| `code/experiments_fdr/diag.py` | `experiments/fdr/diag.py` |  | `a33c625aa9530299e3513167c210a4c331edbb2a02f0bdd5dd13fb44d5af7a7e` | CODE (original, unchanged) |
| `code/experiments_fdr/tests/test_phase3c_leakage.py` | `experiments/fdr/tests/test_phase3c_leakage.py` |  | `4d6396e9b2d3b45b966b3123742da9c73f0633c4b43116db0ec25c1b70689082` | CODE (original, unchanged) |
| `code/experiments_fdr/tests/test_fdr.py` | `experiments/fdr/tests/test_fdr.py` |  | `8aec423abb6ddb09470336303fab9f0d398a815a1ad017bc6e82e2ccbcc8313b` | CODE (original, unchanged) |
| `code/recovery_build_export.py` | `results/extra/phase3c_hidden_state/recovery/build_export.py` |  | `80bcd1377c4ee17e92d7d4839791761119ab90841c1c3d2f5823e7a5584c9d8f` | CODE (this export script) |

## 검증 상태

- [Verified] 파일 존재, 구조, 해시, 체크포인트 로드와 메타데이터 대조, 버그 행의 동일성.
- [Verified] 보고서의 주요 수치와 CSV가 일치한다.
  - 대치 RMSE.
  - MLP 평균: FDR 0.0415, C 0.0557.
  - 원본 창: xz 0.079, x 0.109.
  - x-only 모델: C-x 0.169, FDR-x 0.086.
  - z 제거 수치.
- 재계산 비교는 하지 않았다. 원본이 남아 있어 재계산할 이유가 없었다(Case A).
