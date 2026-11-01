"""Phase 3A: multiplicity c(u) of raw position u as an FDR target inside one training segment of length W=(m-1)tau+K
(anchors t = span..W-1, lags k=0..m-1, target position u = t - k*tau). Also the 'full-multiplicity' range."""
import numpy as np
for m, tau, K in [(8, 1, 64), (8, 5, 64), (8, 5, 16), (4, 10, 64)]:
    span = (m - 1) * tau; W = span + K; c = np.zeros(W, int)
    for t in range(span, W):
        for k in range(m): c[t - k * tau] += 1
    full = np.where(c == m)[0]
    print(f"m={m} tau={tau} K={K} W={W}: c range {c.min()}..{c.max()}, positions with c=m: "
          f"{(full.min(), full.max()) if len(full) else None} ({len(full)}/{W}), first 12 c: {c[:12].tolist()}, last 6: {c[-6:].tolist()}")
