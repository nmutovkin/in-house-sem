"""Discrete-wavenumber reference for a monopole in a fluid-filled borehole.

Geometry (axisymmetric, homogeneous along z): a fluid column r < a, any
sequence of fluid or solid annuli (casing, cement, mud cake, a debonded
fluid micro-annulus, ...), then an unbounded formation. A volume-injection
point source sits on the axis at z = 0 with time function s(t) = d^2Q/dt^2,
as in insem (Source.monopole).

Method (frequency-wavenumber, e^{i(kz - wt)}, complex frequency w + i eps
to smooth guided-mode poles; e.g. Tsang & Rader 1979, Kurkjian & Chang
1986, Tubman et al. 1984):

* fluid layers: p = D_I I0(g r) + D_K K0(g r), u_r = p' / (rho w^2);
* solid layers: u = grad phi + curl curl (z chi), with
  phi = B_I I0(g_p r) + B_K K0(g_p r), chi = C_I I0(g_s r) + C_K K0(g_s r),
  u_r = phi' + i k chi', u_z = i k phi - g_s^2 chi,
  s_rr = -lambda k_p^2 phi + 2 mu (phi'' + i k chi''),
  s_rz = mu (2 i k phi' - (2k^2 - k_s^2) chi'),
  where g_j = sqrt(k^2 - k_j^2) with Re g_j >= 0;
* the formation keeps only K terms (outgoing / decaying);
* the direct field in the column comes from
  e^{i k_f R} / (4 pi R) = (1 / 2 pi) int K0(g_f r) / (2 pi) e^{ikz} dk;
* interfaces: continuity of u_r and s_rr = -p, s_rz = 0 at fluid-solid
  faces, of (u_r, u_z, s_rr, s_rz) at solid-solid faces and (p, u_r) at
  fluid-fluid faces, assembled into one global matrix per k.

I and K terms are normalised at the outer and inner radius of their layer,
which keeps the global matrix well conditioned at large k.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import ive, kve

from .._core import Material

Layer = Tuple[Material, float]  # (material, outer radius)


def layer_matrix_size(layers: Sequence[Layer], formation: Material) -> int:
    """Number of unknowns of the global matrix."""
    n = 1  # reflected field in the fluid column
    for mat, _ in layers[1:]:
        n += 2 if mat.medium == "acoustic" else 4
    return n + (1 if formation.medium == "acoustic" else 2)


def _ifun(g, r, r_norm):
    """I0(g r) / I0(g r_norm) and I1(g r) / I0(g r_norm)."""
    scale = np.exp(np.abs(np.real(g)) * (r - r_norm))
    d = ive(0, g * r_norm)
    return ive(0, g * r) / d * scale, ive(1, g * r) / d * scale


def _kfun(g, r, r_norm):
    """K0(g r) / K0(g r_norm) and K1(g r) / K0(g r_norm)."""
    scale = np.exp(-g * (r - r_norm))
    d = kve(0, g * r_norm)
    return kve(0, g * r) / d * scale, kve(1, g * r) / d * scale


def _basis(kind, g, r, r_norm):
    """(f, f', f'') at r of one normalised radial function."""
    if kind == "I":
        f0, f1 = _ifun(g, r, r_norm)
        return f0, g * f1, g * g * f0 - g / r * f1
    f0, f1 = _kfun(g, r, r_norm)
    return f0, -g * f1, g * g * f0 + g / r * f1


class _LayerFn:
    """Rows (as functions of the layer's unknowns) of the interface quantities."""

    def __init__(self, mat: Material, kinds: List[str], r_in: float, r_out: float, w, k):
        self.mat, self.kinds, self.r_in, self.r_out, self.w, self.k = mat, kinds, r_in, r_out, w, k
        self.fluid = mat.medium == "acoustic"
        self.kp2 = (w / mat.vp) ** 2
        self.gp = np.sqrt(k * k - self.kp2 + 0j)
        if not self.fluid:
            self.ks2 = (w / mat.vs) ** 2
            self.gs = np.sqrt(k * k - self.ks2 + 0j)

    @property
    def size(self):
        return len(self.kinds) * (1 if self.fluid else 2)

    def _norm(self, kind):
        return self.r_out if kind == "I" else self.r_in

    def rows(self, r):
        """dict quantity -> list of coefficient arrays (one per unknown)."""
        k, w, m = self.k, self.w, self.mat
        if self.fluid:
            p, ur = [], []
            for kind in self.kinds:
                f, d1, _ = _basis(kind, self.gp, r, self._norm(kind))
                p.append(f)
                ur.append(d1 / (m.rho * w * w))
            return {"p": p, "ur": ur}
        mu = m.rho * m.vs**2
        lam = m.rho * m.vp**2 - 2 * mu
        out = {"ur": [], "uz": [], "srr": [], "srz": []}
        for kind in self.kinds:  # P potential
            f, d1, d2 = _basis(kind, self.gp, r, self._norm(kind))
            out["ur"].append(d1)
            out["uz"].append(1j * k * f)
            out["srr"].append(-lam * self.kp2 * f + 2 * mu * d2)
            out["srz"].append(2j * mu * k * d1)
        for kind in self.kinds:  # SV potential
            f, d1, d2 = _basis(kind, self.gs, r, self._norm(kind))
            out["ur"].append(1j * k * d1)
            out["uz"].append(-self.gs**2 * f)
            out["srr"].append(2j * mu * k * d2)
            out["srz"].append(-mu * (2 * k * k - self.ks2) * d1)
        return out


def _reflection(layers: Sequence[Layer], formation: Material, w, k):
    """Reflected-field coefficient A(k) of the fluid column, normalised so that
    the column pressure is K0(g_f r) + A I0(g_f r) / I0(g_f a)."""
    fluid, a = layers[0]
    if fluid.medium != "acoustic":
        raise ValueError("the first layer (the borehole) must be a fluid")
    radii = [r for _, r in layers]
    fns = []
    core = _LayerFn(fluid, ["I"], 0.0, a, w, k)
    fns.append(core)
    for (mat, r_out), r_in in zip(layers[1:], radii[:-1]):
        fns.append(_LayerFn(mat, ["I", "K"], r_in, r_out, w, k))
    fns.append(_LayerFn(formation, ["K"], radii[-1], np.inf, w, k))
    n = sum(f.size for f in fns)
    nk = k.size
    M = np.zeros((nk, n, n), complex)
    rhs = np.zeros((nk, n), complex)
    # Direct field at the borehole wall (moved to the right-hand side).
    gf = core.gp
    k0 = kve(0, gf * a) * np.exp(-gf * a)
    k1 = kve(1, gf * a) * np.exp(-gf * a)
    direct = {"p": k0, "ur": -gf * k1 / (fluid.rho * w * w)}
    row = 0
    offset = 0
    for inner, outer, r in zip(fns[:-1], fns[1:], radii):
        ri, ro = inner.rows(r), outer.rows(r)
        o_in, o_out = offset, offset + inner.size
        if inner.fluid and outer.fluid:
            eqs = [("p", "p", 1.0), ("ur", "ur", 1.0)]
        elif inner.fluid or outer.fluid:
            # u_r continuous; s_rr + p = 0; s_rz = 0 (on the solid side).
            eqs = [("ur", "ur", 1.0), ("p", "srr", -1.0) if inner.fluid else ("srr", "p", -1.0), ("srz", None, 0.0)]
        else:
            eqs = [(q, q, 1.0) for q in ("ur", "uz", "srr", "srz")]
        for q_in, q_out, sign in eqs:
            # inner quantity - sign * outer quantity = 0
            if q_out is None:  # s_rz = 0 on the solid side of a fluid-solid face
                coeffs, o = (ro["srz"], o_out) if inner.fluid else (ri["srz"], o_in)
                for j, c in enumerate(coeffs):
                    M[:, row, o + j] = c
                row += 1
                continue
            for j, c in enumerate(ri[q_in]):
                M[:, row, o_in + j] += c
            for j, c in enumerate(ro[q_out]):
                M[:, row, o_out + j] -= sign * c
            if inner is core:
                rhs[:, row] -= direct[q_in]
            row += 1
        offset = o_out
    assert row == n, (row, n)
    return np.linalg.solve(M, rhs[..., None])[..., 0][:, 0]


def borehole_axis_pressure(t, stf, offsets, layers: Sequence[Layer], formation: Material, pad: int = 4,
                           dk_factor: float = 4.0) -> np.ndarray:
    """Pressure (len(t), len(offsets)) on the borehole axis at z = offsets
    for a monopole s(t) = d^2Q/dt^2 on the axis at z = 0.

    layers: [(fluid, a), (material, outer radius), ...]; formation: the
    unbounded outer material. Offsets must be non-zero.
    """
    t = np.asarray(t, float)
    offsets = np.atleast_1d(np.asarray(offsets, float))
    fluid, a = layers[0]
    n = len(t)
    dt = t[1] - t[0]
    m = pad * n
    eps = np.pi / (m * dt)
    tt = np.arange(m) * dt
    s = np.zeros(m)
    s[:n] = stf
    sd = np.fft.rfft(s * np.exp(-eps * tt))
    w = 2 * np.pi * np.fft.rfftfreq(m, dt)
    active = np.abs(sd) > 1e-6 * np.abs(sd).max()
    v_min = min([mat.vs if mat.medium == "elastic" else mat.vp for mat, _ in layers]
                + [formation.vs if formation.medium == "elastic" else formation.vp])
    kmax = 1.5 * w[active].max() / v_min + 15.0 / a  # the reflected field decays like e^{-2 k a}
    dk = eps / (dk_factor * v_min)
    k = np.arange(0.0, kmax + dk, dk)
    taper = np.cos(0.5 * np.pi * np.clip((k - 0.8 * kmax) / (0.2 * kmax), 0, 1)) ** 2
    weights = np.full(k.size, dk)
    weights[0] *= 0.5
    U = np.zeros((len(w), len(offsets)), complex)
    for i in np.nonzero(active)[0]:
        if i == 0:
            continue
        om = w[i] + 1j * eps
        A = _reflection(layers, formation, om, k)
        gf = np.sqrt(k * k - (om / fluid.vp) ** 2 + 0j)
        on_axis = A * np.exp(-np.abs(np.real(gf * a))) / ive(0, gf * a)  # A I0(0) / I0(g_f a)
        # Even in k: (1/2pi) int_-inf^inf F e^{ikz} dk = (1/pi) int_0^inf F cos(kz) dk.
        F = fluid.rho / (2 * np.pi) * on_axis * taper * weights
        U[i] = (F[None, :] * np.cos(np.outer(offsets, k))).sum(axis=1) / np.pi
    reflected = np.array([np.exp(eps * tt) * np.fft.irfft(np.conj(U[:, j]) * sd, m) for j in range(len(offsets))]).T[:n]
    direct = np.array([fluid.rho * np.interp(t - abs(z) / fluid.vp, t, stf, left=0.0) / (4 * np.pi * abs(z))
                       for z in offsets]).T
    return direct + reflected
