# Phase 8 post-freeze change log

Freeze: results/extra/phase8/PROTOCOL_FREEZE_p8.txt (sha256 86243a77e2e806d7a996adb4f47bac17a4963ca863e572c4248ac391ebdf07d2, 2026-10-29 11:16:01 KST).
Every change to a frozen file after this point is recorded here (what, why, when, effect on analyses) and in protocol v2 section 14.

- (none)

## Execution-environment changes (no frozen file changed)
- 2026-10-31 16:2x KST, resume after the server GPU installation (stop 2026-10-29 13:19, at 432/1276 tasks):
  - NVIDIA driver 580.105.08 -> 580.196.07 (logs/gpus_before_shutdown.txt vs logs/gpus_at_resume_20261031.txt).
  - GPU 0 (00000000:49:00.0) and GPU 1 (00000000:4E:00.0) are the same L40S cards in the same PCI order. GPUs 2 and 3 (L40S, C6/CF) were added.
  - The run still uses GPUS=0,1 only (launch_p8.sh unchanged). Effect: tasks finished before 10-29 13:19 ran on driver 580.105.08 and the rest on 580.196.07, on the same two cards. Training randomness is replicate-level, so no analysis change is planned. The driver is documented per task by its start time in logs/sched8.log.

## Frozen-file changes
- 2026-10-31 16:3x KST — run on all four GPUs (user decision). Change made during Gate D, at 432/1276 tasks.
  - experiments/phase8/launch_p8.sh: the GPU list now comes from env P8_GPUS (default "0,1" = frozen behaviour). The labres-check guard checks every listed GPU, and launch.log records GPUS.
    - Old sha256 3055471b0e199bf4eaa6aab66ee8eb95585907d86e1bd3700736bd7cfbab4713. New sha256 fbc2b210603147457da5e674c6d5c4f61cdefd480ee0ea140f9a460c7a0a2fc3.
  - results/notes/phase8_mathematical_protocol_v2.md: a §14 entry was appended for this change.
    - Old sha256 79dfb379e48854b9cf5351e0a32b7f7d8c74b984491f779d4beada8f9df86a86. New sha256 14478cf0f35095cad78a990c50c3cec0454ec7231e8591c351ddf1f337b9a38e.
  - PROTOCOL_FREEZE_p8.txt: those two hash lines were updated, with the old hashes kept as '#' comments. The file was made writable for the edit and set back to 444. Its new sha256 is 98b55c93cfb8dfb5a155309ddb515a57f66fe0fed9ee0827a7faf231a1ff2d8b; the Gate C value was 86243a77…07d2.
  - Restart: the 0,1-GPU run started 16:23 was stopped by SIGTERM, and its unfinished tasks rerun with no failure mark. The restart uses P8_GPUS=0,1,2,3 and NPAR=32 (64 CPUs, OMP_NUM_THREADS=2 per task).
  - Effect on analyses: none. The task graph, seeds, hypotheses, statistics and decision rules are unchanged. All four GPUs are L40S (driver 580.196.07). The GPU used by each task is in logs/sched8.log.
- 2026-10-31 16:2x KST: second launch_p8.sh change, made because the reservation was extended to 2026-11-01 01:16.
  - Problem: labres-check prints an end on another day as "~M/D HH:MM". The guard matched only "~HH:MM", so it refused a valid reservation.
  - Fix: the guard now adds "M/D " to the expected end when END is not today. It was tested against the live labres-check output for GPUs 0–3, and a wrong END is still refused.
  - Hashes: launch_p8.sh old sha256 fbc2b210603147457da5e674c6d5c4f61cdefd480ee0ea140f9a460c7a0a2fc3, new acf27bf6c6b2c0f3955ecbca0cc4b4bf40e564e925ec3c24490bd820fdfa0e7a. PROTOCOL_FREEZE_p8.txt was updated the same way as above and set back to 444; its new sha256 is 52112cbd65044e231b485dc7ac667a5f5a1f51b6c38cc12aabc3c37969701f1c.
  - No effect on analyses: only the launch guard changed.
