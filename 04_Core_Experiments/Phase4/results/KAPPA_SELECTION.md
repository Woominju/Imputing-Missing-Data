# κ selection (2026-10-27 16:17 KST; written BEFORE any coupled-Lorenz training)

Source: `obs_sync.csv`, `obs_local.csv`, `obs_global.csv` (observability.py, CPU, traj 0, dt = 0.05).

| κ | corr(x1,x2) | sync-time fraction (rel. dist < 0.05) | TwoNN dim (6-D state) | x1-only → y2 ambiguity (random pairs ≈ 0.95) | local rank of x1 window | decision |
|---|---|---|---|---|---|---|
| 0 | 0.004 | 0.001 | 4.05 | 0.86 | 3 (s2 unobservable) | **negative control** |
| 0.5 | 0.13 | 0 | 3.97 | 0.78 | 6, σ6 ≈ 0.001 | candidate |
| **1.0** | 0.27 | 0 | 3.98 | 0.78 | 6, σ6 ≈ 0.002, amp_y2 ≈ 230 | **selected κ > 0** |
| 2.0 | 0.68 | 0.085 | 3.66 | 0.56 | 6 | rejected: intermittent synchronisation (16.5 % of time rel. dist < 0.2) |
| 3.0 | 0.86 | 0.28 | 3.57 | 0.20 | 6 | rejected: frequent synchronisation |
| ≥ 4 | 1.00 | ≥ 0.99 | ≈ 2.0 | ≈ 0.04 | — | rejected: complete synchronisation (3-D dynamics) |

Why κ = 1.0:
1. The coupling is real: the x1 window has full local rank 6, which it does not have at κ = 0.
2. There is no synchronisation: the near-sync time fraction is 0 and the TwoNN dimension stays near 4, as at κ = 0.
3. A single view carries only weak, ill-conditioned information about the other subsystem: y2 ambiguity is 0.78 against 0.95 for random pairs, and amp_y2 ≈ 230. Combined views are much better: y2 ambiguity is 0.17–0.19.
4. κ = 1.0 is the strongest coupling that still meets condition 2. That gives the largest single-view vs. combined-view contrast short of synchronisation.

No model results were available or consulted when this choice was made.
