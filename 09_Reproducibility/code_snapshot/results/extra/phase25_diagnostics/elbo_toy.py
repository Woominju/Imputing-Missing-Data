"""Phase 2.5 toy check of the objectives (pure numpy, no training of project models).
x_in carries no sign information (like a z-only view); x_H = +a or -a with prob 1/2 (like x under Gamma).
Latent zeta in R, prior N(0,1), Gaussian q(zeta|x_in) = N(m, s^2), decoder p(x_H|zeta) = N(x_H; R(zeta), sig^2).
Compare, over a grid of (m, s) and two decoders R:
  L_old  = E_{x_H} [ E_q log p(x_H|zeta) ] - KL(q||p)                         (Phase-2 objective)
  L_cond = E_{x_H} [ log E_q p(x_H|zeta) ]                                     (true conditional log-lik of the q-model)
and report the predictive density at the true targets for the L_old optimum and the L_cond optimum."""
import numpy as np
a, sig = 1.0, 0.1
zs, w = np.polynomial.hermite_e.hermegauss(201); w = w / w.sum()       # E over N(0,1)
decoders = {"identity R(z)=z": lambda z: z, "tanh R(z)=a*tanh(5z)": lambda z: a * np.tanh(5 * z)}
logN = lambda x, mu: -0.5 * np.log(2 * np.pi * sig ** 2) - (x - mu) ** 2 / (2 * sig ** 2)
for name, R in decoders.items():
    best_old, best_cond = (-np.inf,), (-np.inf,)
    for m in np.linspace(-1.5, 1.5, 61):
        for s in np.geomspace(1e-3, 3, 60):
            z = m + s * zs
            Lold = np.mean([np.sum(w * logN(x, R(z))) for x in (a, -a)]) - (0.5 * (s ** 2 + m ** 2 - 1) - np.log(s))
            Lcond = np.mean([np.log(np.sum(w * np.exp(logN(x, R(z))))) for x in (a, -a)])
            if Lold > best_old[0]: best_old = (Lold, m, s, Lcond)
            if Lcond > best_cond[0]: best_cond = (Lcond, m, s)
    print(f"[{name}]  argmax L_old: m={best_old[1]:+.3f} s={best_old[2]:.4f} -> predictive mean log-density at +-a = {best_old[3]:.2f}")
    print(f"[{name}]  argmax L_cond: m={best_cond[1]:+.3f} s={best_cond[2]:.4f} -> predictive mean log-density at +-a = {best_cond[0]:.2f}")
print("reference: log-density of the true mixture at +-a =", round(float(np.log(0.5 * np.exp(logN(a, a)) + 0.5 * np.exp(logN(a, -a)))), 2))
