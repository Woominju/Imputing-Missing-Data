# Official FPRM code — provenance

- Paper: "Dynamics-informed machine learning for recovering extensive missing systems dynamics",
  Nature Communications (2026), DOI 10.1038/s41467-026-77922-1
- Code/data record: Figshare DOI 10.6084/m9.figshare.30446765, **version 3**
  (published 2026-05-14T01:39:01Z, modified 2026-05-14T01:40:55Z), licence **CC BY 4.0**
- Live scripts saved with MATLAB R2024b (24.2.0.2712019), per `mwcorePropertiesReleaseInfo.xml` inside the .mlx.
- Downloaded 2026-10-25 (~19:43 KST) from `https://ndownloader.figshare.com/files/<id>` in an earlier Claude Code
  session; copied here 2026-10-27. MD5 of every file equals Figshare's `computed_md5` (`figshare_metadata_v3.json`).
- The data files of the record (EEG, ENSO, exchange, transportation, wind) were not re-downloaded here.

**`original/` is read-only and must not be edited.** File names are the Figshare names (typos included).

| Figshare id | File | MD5 | SHA-256 |
|---|---|---|---|
| 64540893 | FPRM mian code.mlx | 33f9655df4492ec20b184229497ca9da | f2afaae07de0d2c73474be3eaaefb8a8d0bdaac6ea6de9e0a98b485dd8e9260b |
| 64540899 | PhaSpaRecon.m | afb1cde2e407eb9725da6ff428af2267 | c13dbd7dd5dd40f0436f28aaf16d414062660500d1afd14c6bbda151225cad8e |
| 64540896 | multiCoupledLorenz2.m | 4d29fe38dbbf8717cef4366825bb6371 | f07904d93211c597f582187882c47e7c7289ce5cd409eb358ede3b5944a9807e |
| 64540887 | Coupled Loernz system.mlx | 518320e1f692ff5ee761e15d0c8774ff | 98451d648f49e485f4dc9a5eb1f3180e180b081471055ee28572f551b6715f3b |
| 64540890 | Coupled Rossler system.mlx | 65857609bcd122cffc2db69cb3a20741 | d0e13b786b6f1c58f367c4a0176327f8403994da89ad8a313c5fffd190e5c578 |
| 64540884 | 3D ecology system.mlx | c5aa17e7ace9f44b1e22d4611c80d24c | a861293045aa5c0e0611264ab4e11660b7624e2676c2d948a259e410f69b7946 |

`extracted_text/` (derived, for reading without MATLAB):
- `live_scripts_code.txt` — code cells of the four .mlx files (text of `matlab/document.xml`).
- `FPRM_main_saved_output_figure.png` — the only saved output in `FPRM mian code.mlx` (Original vs Predicted
  plot, ~910 test points). No numeric outputs are stored in the live scripts.

No MATLAB/Octave is installed on this server, so none of this code has been executed here.
The Python port lives in `baselines/fprm/` (see `results/notes/fprm_python_port_report.md`).
