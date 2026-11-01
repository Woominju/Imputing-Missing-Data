# Phase 9 post-freeze change log
Freeze: results/extra/phase9/PROTOCOL_FREEZE_p9.txt (sha256 db658aa01aedc2a1bf483cfa838219cd21ceb2fc52f66a5f212c6fc17f3da0af). Every later change to a frozen
file is recorded here AND in protocol section 12. New files (no frozen file changed) are recorded here with their own freeze file.

- 2026-10-31 19:38:58: ADDED experiments/phase9/jena9.py, which implements protocol section 7 (real-data, secondary, descriptive) exactly as specified.
  - Added during the pilot, before any Jena training.
  - Hash in results/extra/phase9/PROTOCOL_FREEZE_p9_jena.txt (444).
  - No frozen file changed.
  - Smoke-tested in a scratch root.
  - Interpretation of the protocol text:
    - SelfV = 5 single-variable AEs with latent 4 each (20 total).
    - The init seed is 9800 + r.
    - Masks: seed_of("p9jena", p, r, split) multiplied by the natural observation indicator.
    - Targets are the artificially masked positions where the truth exists.
