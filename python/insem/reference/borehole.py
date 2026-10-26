"""Discrete-wavenumber references for monopole and dipole sources in a
fluid-filled borehole.

Geometry (homogeneous along z): a fluid column r < a, any sequence of fluid
or solid annuli (casing, cement, mud cake, a debonded fluid micro-annulus,
...), then an unbounded formation. The source sits on the axis at z = 0:

* monopole (``borehole_axis_pressure``): a volume-injection point source
  with time function s(t) = d^2Q/dt^2, as insem's ``Source.monopole``;
* dipole (``borehole_dipole_pressure``): a horizontal point force
  f(t) e_x in the fluid, as insem's ``Source.dipole(0, 0, 1, 0, stf)``;
  the fields vary as cos(theta) (azimuthal order 1) and the result is the
  pressure amplitude P of p = P cos(theta).

Method (frequency-wavenumber, e^{i(kz - wt)}, complex frequency w + i eps
to smooth guided-mode poles; e.g. Tsang & Rader 1979, Kurkjian & Chang
1986, Schmitt 1988). For azimuthal order m:

* fluid layers: p = D_I I_m(g r) + D_K K_m(g r) (times cos(m theta)),
  u_r = p' / (rho w^2);
* solid layers: u = grad phi + curl(z chi) + curl curl(z eta), with
  phi = Phi(r) cos(m theta), eta = H(r) cos(m theta), chi = X(r) sin(m theta)
  (chi only for m = 1), each a combination of I_m and K_m of g_p r (Phi) or
  g_s r (X, H), so that
    U_r = Phi' + m X / r + i k H',
    U_t = -m Phi / r - X' - i k m H / r,
    U_z = i k Phi - g_s^2 H,
    s_rr = -lambda k_p^2 Phi + 2 mu U_r',
    s_rt = mu (-m U_r / r + U_t' - U_t / r),
    s_rz = mu (i k U_r + U_z'),
  where g_j = sqrt(k^2 - k_j^2) with Re g_j >= 0 and u_theta = U_t sin(m theta);
* the formation keeps only K terms (outgoing / decaying);
* the direct field in the column comes from
  e^{i k_f R} / (4 pi R) = (1 / 2 pi) int K0(g_f r) / (2 pi) e^{ikz} dk;
  for the dipole p = -f d_x of it, i.e. f g_f K1(g_f r) / (2 pi) cos(theta);
* interfaces: continuity of u_r and s_rr = -p, s_rt = s_rz = 0 at
  fluid-solid faces, of all displacement and traction components at
  solid-solid faces and of (p, u_r) at fluid-fluid faces, assembled into
  one global matrix per k.

I and K terms are normalised at the outer and inner radius of their layer,
which keeps the global matrix well conditioned at large k.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import ive, kve

from .._core import Material

Layer = Tuple[Material, float]  # (material, outer radius)


def _potentials(mat: Material, order: int) -> int:
    """Number of potentials of a layer (fluid: 1; solid: 2 for m = 0, 3 for m = 1)."""
    return 1 if mat.medium == "acoustic" else 2 + order


def layer_matrix_size(layers: Sequence[Layer], formation: Material, order: int = 0) -> int:
    """Number of unknowns of the global matrix."""
    n = 1  # reflected field in the fluid column
    for mat, _ in layers[1:]:
        n += 2 * _potentials(mat, order)
    return n + _potentials(formation, order)


def _basis(order, kind, g, r, r_norm):
    """(f, f', f'') at r of I_m(g r) / I_m(g r_norm) or K_m(g r) / K_m(g r_norm), m = order."""
    if kind == "I":
        scale = np.exp(np.abs(np.real(g)) * (r - r_norm)) / ive(order, g * r_norm)
        f, lower = ive(order, g * r) * scale, ive(order - 1, g * r) * scale if order else ive(1, g * r) * scale
        d1 = g * lower - order * f / r if order else g * lower  # I_m' = I_{m-1} - m I_m / x; I_0' = I_1
    else:
        scale = np.exp(-g * (r - r_norm)) / kve(order, g * r_norm)
        f, lower = kve(order, g * r) * scale, kve(order - 1, g * r) * scale if order else kve(1, g * r) * scale
        d1 = -g * lower - order * f / r if order else -g * lower  # K_m' = -K_{m-1} - m K_m / x; K_0' = -K_1
    d2 = -d1 / r + (g * g + order * order / (r * r)) * f
    return f, d1, d2


class _LayerFn:
    """Rows (as functions of the layer's unknowns) of the interface quantities."""

    def __init__(self, mat: Material, kinds: List[str], r_in: float, r_out: float, w, k, order: int):
        self.mat, self.kinds, self.r_in, self.r_out, self.w, self.k = mat, kinds, r_in, r_out, w, k
        self.order = order
        self.fluid = mat.medium == "acoustic"
        self.kp2 = (w / mat.vp) ** 2
        self.gp = np.sqrt(k * k - self.kp2 + 0j)
        if not self.fluid:
            self.ks2 = (w / mat.vs) ** 2
            self.gs = np.sqrt(k * k - self.ks2 + 0j)

    @property
    def size(self):
        return len(self.kinds) * _potentials(self.mat, self.order)

    def _norm(self, kind):
        return self.r_out if kind == "I" else self.r_in

    def rows(self, r):
        """dict quantity -> list of coefficient arrays (one per unknown)."""
        k, w, m, n = self.k, self.w, self.mat, self.order
        if self.fluid:
            p, ur = [], []
            for kind in self.kinds:
                f, d1, _ = _basis(n, kind, self.gp, r, self._norm(kind))
                p.append(f)
                ur.append(d1 / (m.rho * w * w))
            return {"p": p, "ur": ur}
        mu = m.rho * m.vs**2
        lam = m.rho * m.vp**2 - 2 * mu
        names = ("ur", "ut", "uz", "srr", "srt", "srz")
        out = {q: [] for q in names}
        ik = 1j * k
        potentials = ["phi", "h"] if n == 0 else ["phi", "x", "h"]
        for pot in potentials:
            g = self.gp if pot == "phi" else self.gs
            for kind in self.kinds:
                f, d1, d2 = _basis(n, kind, g, r, self._norm(kind))
                # displacement amplitudes and their r derivatives
                if pot == "phi":
                    ur, ut, uz = d1, -n * f / r, ik * f
                    ur1, ut1, uz1 = d2, -n * (d1 / r - f / r**2), ik * d1
                    div = -self.kp2 * f
                elif pot == "x":
                    ur, ut, uz = n * f / r, -d1, 0 * f
                    ur1, ut1, uz1 = n * (d1 / r - f / r**2), -d2, 0 * f
                    div = 0 * f
                else:
                    ur, ut, uz = ik * d1, -ik * n * f / r, -self.gs**2 * f
                    ur1, ut1, uz1 = ik * d2, -ik * n * (d1 / r - f / r**2), -self.gs**2 * d1
                    div = 0 * f
                vals = {
                    "ur": ur, "ut": ut, "uz": uz,
                    "srr": lam * div + 2 * mu * ur1,
                    "srt": mu * (-n * ur / r + ut1 - ut / r),
                    "srz": mu * (ik * ur + uz1),
                }
                for q in names:
                    out[q].append(vals[q])
        return out


def _system(layers: Sequence[Layer], formation: Material, w, k, order: int = 0):
    """Global matrix M (len(k), n, n) and right-hand side (direct field) of
    the interface conditions; the first unknown is the reflected-field
    coefficient of the fluid column."""
    fluid, a = layers[0]
    if fluid.medium != "acoustic":
        raise ValueError("the first layer (the borehole) must be a fluid")
    radii = [r for _, r in layers]
    fns = []
    core = _LayerFn(fluid, ["I"], 0.0, a, w, k, order)
    fns.append(core)
    for (mat, r_out), r_in in zip(layers[1:], radii[:-1]):
        fns.append(_LayerFn(mat, ["I", "K"], r_in, r_out, w, k, order))
    fns.append(_LayerFn(formation, ["K"], radii[-1], np.inf, w, k, order))
    n = sum(f.size for f in fns)
    nk = k.size
    M = np.zeros((nk, n, n), complex)
    rhs = np.zeros((nk, n), complex)
    # Direct field K_m(g_f r) at the borehole wall (moved to the right-hand side).
    gf = core.gp
    kd = kve(order, gf * a) * np.exp(-gf * a)
    kl = kve(order - 1 if order else 1, gf * a) * np.exp(-gf * a)
    dkd = -gf * kl - order * kd / a
    direct = {"p": kd, "ur": dkd / (fluid.rho * w * w)}
    solid_eqs = ["ur", "uz", "srr", "srz"] if order == 0 else ["ur", "ut", "uz", "srr", "srt", "srz"]
    free = ["srz"] if order == 0 else ["srt", "srz"]  # shear tractions on a fluid-solid face
    row = 0
    offset = 0
    for inner, outer, r in zip(fns[:-1], fns[1:], radii):
        ri, ro = inner.rows(r), outer.rows(r)
        o_in, o_out = offset, offset + inner.size
        if inner.fluid and outer.fluid:
            eqs = [("p", "p", 1.0), ("ur", "ur", 1.0)]
        elif inner.fluid or outer.fluid:
            # u_r continuous; s_rr + p = 0; zero shear tractions (on the solid side).
            eqs = [("ur", "ur", 1.0), ("p", "srr", -1.0) if inner.fluid else ("srr", "p", -1.0)]
            eqs += [(q, None, 0.0) for q in free]
        else:
            eqs = [(q, q, 1.0) for q in solid_eqs]
        for q_in, q_out, sign in eqs:
            # inner quantity - sign * outer quantity = 0
            if q_out is None:  # a shear traction vanishes on the solid side of a fluid-solid face
                coeffs, o = (ro[q_in], o_out) if inner.fluid else (ri[q_in], o_in)
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
    return M, rhs


def _reflection(layers: Sequence[Layer], formation: Material, w, k, order: int = 0):
    """Reflected-field coefficient A(k) of the fluid column, normalised so that
    the column pressure is K_m(g_f r) + A I_m(g_f r) / I_m(g_f a)."""
    M, rhs = _system(layers, formation, w, k, order)
    return np.linalg.solve(M, rhs[..., None])[..., 0][:, 0]


def borehole_mode_slowness(freqs, layers: Sequence[Layer], formation: Material, order: int = 1,
                           s_max: float = None, n_scan: int = 400) -> np.ndarray:
    """Phase slowness (s/m) of the slowest guided mode at each frequency (Hz):
    the Stoneley wave for azimuthal order 0, the flexural wave for order 1.

    Modal root-finding of the borehole dispersion relation det M(w, k) = 0
    (the global matrix of the discrete-wavenumber reference at real w): the
    smallest singular value of the row- and column-scaled matrix is scanned
    over slowness, from just above the formation shear slowness (the mode
    must be trapped, so the formation must be a solid slower than the mode)
    to s_max (default 3 / vs), on a grid that is geometric in s vs - 1 (at
    low frequency the flexural wave approaches 1 / vs), and its slowest
    local minimum is refined.
    """
    from scipy.optimize import minimize_scalar

    if formation.medium != "elastic":
        raise ValueError("guided-mode slowness needs a solid formation")
    s_hi = s_max if s_max is not None else 3.0 / formation.vs

    def sigma(s, w):
        M, _ = _system(layers, formation, w, np.atleast_1d(w * s), order)
        M = M / np.abs(M).max(axis=2, keepdims=True)
        M = M / np.abs(M).max(axis=1, keepdims=True)
        return np.linalg.svd(M, compute_uv=False)[:, -1]

    grid = (1.0 + np.geomspace(1e-7, s_hi * formation.vs - 1.0, n_scan)) / formation.vs
    out = []
    for f in np.atleast_1d(np.asarray(freqs, float)):
        w = 2 * np.pi * f
        sv = np.array([sigma(s, w)[0] for s in grid])
        minima = [i for i in range(1, n_scan - 1) if sv[i] <= sv[i - 1] and sv[i] <= sv[i + 1]]
        if not minima:
            raise ValueError(f"no trapped guided mode between {grid[0]:.6g} and {s_hi:.4g} s/m at {f:g} Hz "
                             "(at low frequency the flexural slowness tends to 1 / vs, closer than the scan)")
        i = max(minima)  # the slowest mode
        res = minimize_scalar(lambda s: sigma(s, w)[0], bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                              options={"xatol": 1e-12})
        out.append(res.x)
    return np.array(out)


def _spectrum(t, stf, pad):
    """Damped spectrum of the source time function: (sd, w, eps, m, tt)."""
    n = len(t)
    dt = t[1] - t[0]
    m = pad * n
    eps = np.pi / (m * dt)
    tt = np.arange(m) * dt
    s = np.zeros(m)
    s[:n] = stf
    sd = np.fft.rfft(s * np.exp(-eps * tt))
    w = 2 * np.pi * np.fft.rfftfreq(m, dt)
    return sd, w, eps, m, tt


def _slowest(layers, formation):
    return min([mat.vs if mat.medium == "elastic" else mat.vp for mat, _ in layers]
               + [formation.vs if formation.medium == "elastic" else formation.vp])


def _wavenumbers(w_max, v_min, decay_length, eps, dk_factor):
    """Wavenumber samples, cosine taper and trapezoid weights on [0, kmax].
    The reflected field decays like e^{-k decay_length} at large k."""
    kmax = 1.5 * w_max / v_min + 30.0 / decay_length
    dk = eps / (dk_factor * v_min)
    k = np.arange(0.0, kmax + dk, dk)
    taper = np.cos(0.5 * np.pi * np.clip((k - 0.8 * kmax) / (0.2 * kmax), 0, 1)) ** 2
    weights = np.full(k.size, dk)
    weights[0] *= 0.5
    return k, taper * weights


def _to_time(U, sd, eps, m, tt, n):
    """Time series (n, n_receivers) from responses U[w, receiver] (e^{-iwt} convention)."""
    return np.array([np.exp(eps * tt) * np.fft.irfft(np.conj(U[:, j]) * sd, m)
                     for j in range(U.shape[1])]).T[:n]


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
    sd, w, eps, m, tt = _spectrum(t, stf, pad)
    active = np.abs(sd) > 1e-6 * np.abs(sd).max()
    k, kw = _wavenumbers(w[active].max(), _slowest(layers, formation), 2 * a, eps, dk_factor)
    U = np.zeros((len(w), len(offsets)), complex)
    for i in np.nonzero(active)[0]:
        if i == 0:
            continue
        om = w[i] + 1j * eps
        A = _reflection(layers, formation, om, k)
        gf = np.sqrt(k * k - (om / fluid.vp) ** 2 + 0j)
        on_axis = A * np.exp(-np.abs(np.real(gf * a))) / ive(0, gf * a)  # A I0(0) / I0(g_f a)
        # Even in k: (1/2pi) int_-inf^inf F e^{ikz} dk = (1/pi) int_0^inf F cos(kz) dk.
        F = fluid.rho / (2 * np.pi) * on_axis * kw
        U[i] = (F[None, :] * np.cos(np.outer(offsets, k))).sum(axis=1) / np.pi
    reflected = _to_time(U, sd, eps, m, tt, n)
    direct = np.array([fluid.rho * np.interp(t - abs(z) / fluid.vp, t, stf, left=0.0) / (4 * np.pi * abs(z))
                       for z in offsets]).T
    return direct + reflected


def borehole_dipole_pressure(t, stf, offsets, radius: float, layers: Sequence[Layer], formation: Material,
                             pad: int = 4, dk_factor: float = 4.0) -> np.ndarray:
    """Pressure amplitude P (len(t), len(offsets)), p = P cos(theta), at
    (r = radius, z = offsets) for a horizontal point force stf(t) e_x on the
    borehole axis at z = 0 (a dipole source).

    layers: [(fluid, a), (material, outer radius), ...]; formation: the
    unbounded outer material. 0 < radius < a (P vanishes on the axis).
    """
    t = np.asarray(t, float)
    offsets = np.atleast_1d(np.asarray(offsets, float))
    fluid, a = layers[0]
    if not 0.0 < radius < a:
        raise ValueError(f"receiver radius {radius} must be inside the fluid column (0, {a})")
    n = len(t)
    sd, w, eps, m, tt = _spectrum(t, stf, pad)
    active = np.abs(sd) > 1e-6 * np.abs(sd).max()
    # Reflected field at r: K1(g a) I1(g r) / I1(g a) ~ e^{-k (2a - r)}.
    k, kw = _wavenumbers(w[active].max(), _slowest(layers, formation), 2 * a - radius, eps, dk_factor)
    R = np.hypot(radius, offsets)
    U = np.zeros((len(w), len(offsets)), complex)
    for i in np.nonzero(active)[0]:
        if i == 0:
            continue
        om = w[i] + 1j * eps
        A = _reflection(layers, formation, om, k, order=1)
        gf = np.sqrt(k * k - (om / fluid.vp) ** 2 + 0j)
        at_r, _, _ = _basis(1, "I", gf, radius, a)  # I1(g r) / I1(g a)
        F = gf / (2 * np.pi) * A * at_r * kw
        reflected = (F[None, :] * np.cos(np.outer(offsets, k))).sum(axis=1) / np.pi
        # Direct field -d_x [e^{i k_f R} / (4 pi R)] = -g'(R) r / R.
        kf = om / fluid.vp
        g = np.exp(1j * kf * R) / (4 * np.pi * R)
        direct = -g * (1j * kf - 1 / R) * radius / R
        U[i] = reflected + direct
    return _to_time(U, sd, eps, m, tt, n)
