"""Analytic 2D reference solutions for the verification tests.

Frequency-domain Green's functions are written for numpy's FFT convention
(fields ~ e^{+i w t}), where outgoing waves use H^(2). The scalar Green's
function of (lap + k^2) g = -delta is then g = -(i/4) H0^(2)(k r).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import hankel2


def _spectrum(t: np.ndarray, stf: np.ndarray, pad: int = 8):
    n = len(t)
    npad = pad * n
    s = np.zeros(npad)
    s[:n] = stf
    w = 2 * np.pi * np.fft.rfftfreq(npad, t[1] - t[0])
    return np.fft.rfft(s), w, npad


def _scalar_green(k: np.ndarray, r: float):
    """g, g', g'' of g = -(i/4) H0^(2)(k r) with respect to r."""
    kr = k * r
    g = -0.25j * hankel2(0, kr)
    gp = 0.25j * k * hankel2(1, kr)
    gpp = 0.25j * k * k * (hankel2(0, kr) - hankel2(1, kr) / kr)
    return g, gp, gpp


def acoustic_pressure(t, stf, x, z, c, rho):
    """Pressure of a monopole (1/kappa) p_tt = div(grad p / rho) + stf(t) delta at the origin."""
    S, w, npad = _spectrum(t, stf)
    r = np.hypot(x, z)
    G = np.zeros_like(S)
    G[1:] = _scalar_green(w[1:] / c, r)[0]
    return np.fft.irfft(rho * G * S, npad)[: len(t)]


def acoustic_particle_velocity(t, stf, x, z, c, rho):
    """Particle velocity v = -(1/rho) int grad p dt for the same source: (n, 2)."""
    S, w, npad = _spectrum(t, stf)
    r = np.hypot(x, z)
    H = np.zeros_like(S)
    H[1:] = -hankel2(1, w[1:] * r / c) / (4 * c)
    return np.array([np.fft.irfft(H * S * e, npad)[: len(t)] for e in (x / r, z / r)]).T


def elastic_green(w, x, z, vp, vs, rho):
    """Plane-strain Green's tensor G_ij(w) (2, 2, nw) of rho u_tt = div sigma + e_j delta.

    G = (1/mu) [g_s I + (1/k_s^2) grad grad (g_s - g_p)].
    """
    mu = rho * vs * vs
    r = np.hypot(x, z)
    e = np.array([x / r, z / r])
    gs, gsp, gspp = _scalar_green(w / vs, r)
    gq, gqp, gqpp = _scalar_green(w / vp, r)
    G = np.zeros((2, 2, w.size), complex)
    for i in range(2):
        for j in range(2):
            d = 1.0 if i == j else 0.0
            hs = gspp * e[i] * e[j] + gsp / r * (d - e[i] * e[j])
            hp = gqpp * e[i] * e[j] + gqp / r * (d - e[i] * e[j])
            G[i, j] = (gs * d + (hs - hp) / (w / vs) ** 2) / mu
    return G


def elastic_displacement(t, stf, x, z, vp, vs, rho, force=None, moment=None):
    """Displacement (n, 2) at (x, z) of a point force (fx, fz) or moment tensor
    ((mxx, mxz), (mxz, mzz)) at the origin with time function stf."""
    S, w, npad = _spectrum(t, stf)
    W = w.copy()
    W[0] = 1.0  # avoid the k = 0 singularity; the DC bin is zeroed below
    if force is not None:
        G = elastic_green(W, x, z, vp, vs, rho)
        U = G[:, 0] * force[0] + G[:, 1] * force[1]
    else:
        # u_i = -M_jk d_k G_ij (source at the origin), derivatives by central differences.
        h = 1e-4 * np.hypot(x, z)
        dG = [
            (elastic_green(W, x + h, z, vp, vs, rho) - elastic_green(W, x - h, z, vp, vs, rho)) / (2 * h),
            (elastic_green(W, x, z + h, vp, vs, rho) - elastic_green(W, x, z - h, vp, vs, rho)) / (2 * h),
        ]
        M = np.asarray(moment, dtype=float)
        U = -sum(M[j, k] * dG[k][:, j] for j in range(2) for k in range(2))
    U[:, 0] = 0.0
    return np.array([np.fft.irfft(U[i] * S, npad)[: len(t)] for i in range(2)]).T


def rayleigh_speed(vp: float, vs: float) -> float:
    """Rayleigh wave speed of an isotropic half-space."""
    k2 = (vs / vp) ** 2
    f = lambda x: (2 - x * x) ** 2 - 4 * np.sqrt(1 - k2 * x * x) * np.sqrt(1 - x * x)
    return brentq(f, 0.5, 0.9999) * vs


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def lamb_surface(t, stf, x, vp, vs, rho, force=1.0, pad=4):
    """Lamb's problem: surface displacement (n, 2) at offset x of a vertical
    line load -force e_z delta(x) stf(t) on the free surface of the
    half-space z < 0.

    Wavenumber integration with complex frequency w + i eps (eps = pi / T):
    with potentials decaying into z < 0 and a traction-free surface,
        u_z(k, w) = F nu_p k_s^2 / (mu R),
        u_x(k, w) = -i k F (2k^2 - k_s^2 - 2 nu_p nu_s) / (mu R),
        R = (2k^2 - k_s^2)^2 - 4 k^2 nu_p nu_s,  nu_j = sqrt(k^2 - k_j^2),
    in the e^{i(kx - wt)} convention. The k step resolves the Rayleigh pole
    (width ~ eps / c_R); the result does not change with a 4x finer step.
    """
    n = len(t)
    dt = t[1] - t[0]
    m = pad * n
    eps = np.pi / (m * dt)
    tt = np.arange(m) * dt
    s = np.zeros(m)
    s[:n] = stf
    sd = np.fft.rfft(s * np.exp(-eps * tt))
    w = 2 * np.pi * np.fft.rfftfreq(m, dt)
    mu = rho * vs * vs
    active = np.abs(sd) > 1e-6 * np.abs(sd).max()
    kmax = 6 * w[active].max() / vs + 10.0 / max(abs(x), 1e-3)
    dk = eps / (10 * vs)
    k = np.arange(-kmax, kmax + dk, dk)
    taper = np.cos(0.5 * np.pi * np.clip((np.abs(k) - 0.8 * kmax) / (0.2 * kmax), 0, 1)) ** 2
    phase = np.exp(1j * k * x) * taper * dk / (2 * np.pi)
    ux = np.zeros(len(w), complex)
    uz = np.zeros(len(w), complex)
    for i in np.nonzero(active)[0]:
        if i == 0:
            continue
        om = w[i] + 1j * eps
        kp2, ks2 = (om / vp) ** 2, (om / vs) ** 2
        nup, nus = np.sqrt(k * k - kp2), np.sqrt(k * k - ks2)
        r = (2 * k * k - ks2) ** 2 - 4 * k * k * nup * nus
        uz[i] = np.sum(force * nup * ks2 / (mu * r) * phase)
        ux[i] = np.sum(-1j * k * force * (2 * k * k - ks2 - 2 * nup * nus) / (mu * r) * phase)
    # u(t) = e^{eps t} irfft(conj(U) * spectrum of s e^{-eps t})
    return np.array([(np.exp(eps * tt) * np.fft.irfft(np.conj(u) * sd, m))[:n] for u in (ux, uz)]).T
