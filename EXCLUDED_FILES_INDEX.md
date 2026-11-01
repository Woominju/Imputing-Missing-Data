# 제외 자료 색인

원본은 서버에 모두 보존돼 있으며 삭제하지 않았다. 경로는 프로젝트 루트 기준이다.

| 분류 | 제외 자료 | 크기 | 이유 | 확보 또는 재생성 방법 |
|---|---|---|---|---|
| 재생성 가능한 시뮬레이션 | `results/extra/phase{5,6,7,8,9}/data/` 궤적 npz | 약 1.7 GB | 생성기와 seed 규칙으로 결정적으로 재생성된다 | `09_Reproducibility/reproduction_guide.md`. Phase 8/9 해시는 각 번들 FULL_DATA_LIST |
| 학습 산출물 | `results/extra/phase*/runs/ckpt/`, Phase 3–4 checkpoint | 약 1.4 GB | 크고 재학습 가능 | 학습 코드는 `code_snapshot`에 있다. 해시는 고정 파일과 FULL_DATA_LIST |
| Phase 8 oracle | `results/extra/phase8/oracle/*.pt` | 905 MB | 대용량 | `oracle8.py`로 재학습. 해시는 `PROTOCOL_FREEZE_p9.txt` 섹션 D(raw/imp)와 Phase 8 FULL_DATA_LIST |
| Jacobian 산출물 | `results/extra/phase6/jac`, `phase7/jac` | 각 186 MB | 대용량. 결론은 보고서에 요약돼 있다 | `jac6.py`, `jac7.py` |
| Phase 6 S1R 원시 산출물 | `results/extra/phase6/s1r/`(결과 CSV 외) | 315 MB | 대용량 | `s1r_results.csv`는 포함했다 |
| 대용량 중간 자료 | `results/extra/aim_autonomous_20261027`(AUTO_RESULTS.md 외), `phase4_coupled_pilot` npz, `phase3d2_lag_weight` npz/pt, `aim_rescue`/`aim_latent`/`aim_crossview` gz, `expA–D` | 0.2 GB 내외 | 요약 CSV와 보고서를 포함했다. aim_* 10-25 분석은 전용 보고서가 없다(UNVERIFIED) | 원본 경로 |
| 중복 | `results/audit/*/` 압축 해제 폴더, `results/review_packages/*_documents.zip` 중복본 | 95 MB | ZIP과 동일 | ZIP을 포함했다 |
| 제3자 논문 | `third_party/fprm_paper/*.pdf` | — | PROVENANCE: "do not redistribute" | DOI 10.1038/s41467-026-77922-1 |
| 실제 자료(출처·라이선스 미기록) | `Data/*.xlsx`, `Data/Wind speed.csv`(EEG, ENSO, Exchange, Transportation, Wind), `Data/external/{jena,beijing,appliances}` | — | 재배포 조건이 기록되지 않았다. FPRM 자료 5종은 Figshare 10.6084/m9.figshare.30446765(CC BY 4.0)와 MD5가 일치하지만, 이것은 추정이다 | 원 출처에서 받는다(FPRM Figshare. Jena와 Beijing은 `results/notes/phase5_external_dataset_audit.md` 참조) |
| 무관하거나 개인 자료 | `Data/external/beijing/raw/2A2478DC-….JPG`(개인 사진), `raw/{data,test}.csv`(무관한 주가) | — | 연구와 무관하며 개인정보 위험이 있다 | 포함하지 않는다 |
| 실행 환경 | `.pydeps/`, `__pycache__`, lock 파일, GPU job 로그, 서버 포털 호스트명이 든 로그 | — | 재현에 불필요하다 | `requirements.txt` |
| AIM과 무관 | `configs/overnight_watchdog.sh`가 가리키는 존재하지 않는 `experiments/extra/run_exp_a.py`(죽은 참조) | — | 파일 없음 | — |

**포함한 부정적 결과(제외하지 않음)**
- Phase 3C 무효화와 3D-1 정정
- Phase 3D-2 trade-off
- Phase 4 OOD 정정
- Phase 5 H2 미지지와 정오표 E6–E8
- Phase 6 κ = 0 대조 실패, G3 실패, 정정(§A5)
- Phase 7 κ = 0 동일 효과
- Phase 8 IIb-1 미검출, Fact8 우위, F1–F11
- Phase 9 P-1 실패, 학습 발산, Jena 결과

**포함한 제3자 코드:** `06_Baseline_Comparisons/FPRM/fprm_official/`(Figshare v3, **CC BY 4.0**; `PROVENANCE.md`에 출처 표기)
