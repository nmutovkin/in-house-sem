"""Dipole and monopole sonic logging in a slow formation (axisymmetric).

    python examples/borehole/dipole.py [output_dir] [--no-reference]

In a slow formation (shear slower than the borehole fluid) a monopole
excites no refracted shear wave, so its array cannot measure the shear
slowness; a dipole excites the flexural wave, whose slowness tends to the
shear slowness at low frequency. This example runs both on a 10 cm borehole
(formation vp 2500, vs 1000 m/s):

* dipole: azimuthal order 1, a 3 kHz horizontal point force on the axis,
  hydrophones at half the borehole radius (the cos(theta) pressure
  amplitude), compared with the discrete-wavenumber reference;
* monopole: azimuthal order 0, 5 kHz, hydrophones on the axis.

It plots wavefield snapshots, the dipole waveforms against the reference,
the slowness-time coherence of both arrays and the flexural dispersion of
the dipole array against modal root-finding. Needs matplotlib and scipy;
takes a few minutes.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import insem
from insem import mesh, post
from insem.reference import borehole_dipole_pressure, borehole_mode_slowness

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(args[0]) if args else Path("dipole_out")
out.mkdir(parents=True, exist_ok=True)
with_reference = "--no-reference" not in sys.argv
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

MUD = insem.Material.acoustic(1500.0, 1000.0)
FORMATION = insem.Material.elastic(2500.0, 1000.0, 2200.0)
A = 0.1
OFFSETS = np.round(np.linspace(1.0, 2.4, 8), 3)
DURATION = 3.6e-3
CASES = {  # name: (azimuthal order, source, f0, receiver radius)
    "dipole": (1, lambda stf: insem.Source.dipole(0.0, 0.0, 1.0, 0.0, stf), 3000.0, 0.5 * A),
    "monopole": (0, lambda stf: insem.Source.monopole(0.0, 0.0, stf), 5000.0, 0.0),
}


def build(order, source, f0, r_rec):
    # The outer boundary is far enough that the formation P wave it reflects
    # (a few % with first-order Stacey) arrives after the recording window.
    b = mesh.borehole((-1.5, 4.5), [("mud", A)], 4.5, dz=0.08, dr=0.3)
    sim = insem.Simulation(b.mesh, insem.Discretization(order=4, f_max=7500.0, azimuthal_order=order))
    sim.set_material(b.regions["mud"], MUD)
    sim.set_material(b.regions["formation"], FORMATION)
    for name in ("bottom", "outer", "top"):
        sim.set_boundary(b.boundaries[name], "absorbing")
    sim.add_source(source(insem.ricker(f0)))
    for z in OFFSETS:
        sim.add_receiver(f"z{z}", r_rec, float(z), ["pressure"])
    return sim


runs = {}
for name, (order, source, f0, r_rec) in CASES.items():
    sim = build(order, source, f0, r_rec)
    sim.set_time(duration=DURATION, record_every=4)
    dt = sim.solver.dt
    sim.set_time(duration=DURATION, record_every=4, snapshot_every=int(round(0.1e-3 / dt)),
                 snapshot_quantities=["displacement", "pressure"])
    s = sim.solver
    t0 = time.time()
    sim.run()
    print(f"{name}: {sim.dofs.n_elements} elements, dt {s.dt:.2e} s, {s.n_steps} steps, {time.time() - t0:.0f} s")
    sim.write_results(str(out / f"{name}_results.h5"))
    data = np.array([s.trace(f"z{z}", "pressure")[:, 0] for z in OFFSETS])
    ref = None
    if with_reference and name == "dipole":
        t0 = time.time()
        ref = borehole_dipole_pressure(s.times, insem.ricker(f0)(s.times), OFFSETS, r_rec, [(MUD, A)], FORMATION).T
        err = [np.linalg.norm(a - c) / np.linalg.norm(c) for a, c in zip(data, ref)]
        print(f"  reference {time.time() - t0:.0f} s; SEM vs reference: {100 * min(err):.2f}-{100 * max(err):.2f} %")
    runs[name] = dict(sim=sim, t=s.times, data=data, ref=ref, f0=f0, order=order, r_rec=r_rec)


# ------------------------------------------------------------ 1. snapshots
def tris(cells):
    return np.concatenate([cells[:, [0, 1, 2]], cells[:, [0, 2, 3]]])


fig, axes = plt.subplots(2, 4, figsize=(11, 7.5), sharey=True, layout="constrained")
for row, (name, run) in zip(axes, runs.items()):
    s = run["sim"].solver
    ts = s.snapshot_times
    pa, ue = s.snapshot("acoustic", "pressure")[:, 0], s.snapshot("elastic", "displacement")
    pts_a, cells_a = s.snapshot_points("acoustic"), s.snapshot_cells("acoustic")
    pts_e, cells_e = s.snapshot_points("elastic"), s.snapshot_cells("elastic")
    umag = np.hypot(ue[:, 0], ue[:, -1])  # in the plane theta = 0: (u_r, u_z)
    # In the x-z plane the left half is theta = pi: p = P cos(pi) = -P for the
    # dipole (antisymmetric), p = P for the monopole.
    mirror = -1.0 if run["order"] == 1 else 1.0
    for ax, t_want in zip(row, (0.6e-3, 1.2e-3, 2.0e-3, 3.0e-3)):
        k = int(np.argmin(np.abs(ts - t_want)))
        pmax = 0.3 * np.abs(pa[k]).max()
        umax = 0.5 * umag[k].max()
        for sign in (1, -1):
            ax.tripcolor(sign * pts_e[0], pts_e[1], tris(cells_e), umag[k], cmap="Greys", vmin=0, vmax=umax,
                         shading="gouraud")
            ax.tripcolor(sign * pts_a[0], pts_a[1], tris(cells_a), (mirror if sign < 0 else 1.0) * pa[k],
                         cmap="RdBu_r", vmin=-pmax, vmax=pmax, shading="gouraud")
        ax.axvline(A, color="k", lw=0.3); ax.axvline(-A, color="k", lw=0.3)
        ax.plot(np.full_like(OFFSETS, run["r_rec"]), OFFSETS, "v", color="gold", mec="k", ms=4)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.0, 3.2)
        ax.set_title(f"{name}, t = {ts[k] * 1e3:.2f} ms")
        ax.set_xlabel("x (m), plane theta = 0 / pi")
    row[0].set_ylabel("z (m)")
fig.suptitle("slow formation: pressure in the mud (red/blue), |displacement| in the formation (grey)")
fig.savefig(out / "fig1_dipole_snapshots.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------------ 2. dipole waveforms
run = runs["dipole"]
t, data, ref = run["t"], run["data"], run["ref"]
fig, ax = plt.subplots(figsize=(6.5, 5))
scale = 0.45 * (OFFSETS[1] - OFFSETS[0]) / np.abs(data).max()
for j, z in enumerate(OFFSETS):
    if ref is not None:
        ax.plot(t * 1e3, z + scale * ref[j], color="0.6", lw=2.5, label="reference (DWM)" if j == 0 else None)
    ax.plot(t * 1e3, z + scale * data[j], "k", lw=0.8, label="SEM" if j == 0 else None)
for slow, label in ((1 / FORMATION.vp, "P"), (1 / FORMATION.vs, "S"), (1 / MUD.vp, "mud")):
    ax.plot((1.2 / run["f0"] + slow * OFFSETS) * 1e3, OFFSETS, "--", lw=0.8, color="C3")
    ax.text((1.2 / run["f0"] + slow * OFFSETS[-1]) * 1e3, OFFSETS[-1] + 0.12, label, color="C3")
ax.set_title(f"dipole: pressure amplitude at r = {run['r_rec']:.2f} m")
ax.set_xlabel("time (ms)"); ax.set_ylabel("receiver offset (m)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(out / "fig2_dipole_waveforms.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------------ 3. STC
grid = np.linspace(200e-6, 1400e-6, 601)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (name, run) in zip(axes, runs.items()):
    t, data = run["t"], run["data"]
    window = 0.5e-3 if name == "dipole" else 0.25e-3
    r = post.stc(data, OFFSETS, t[1] - t[0], grid, window=window, times=np.arange(0, 2.6e-3, 5e-6))
    im = ax.pcolormesh(r.time * 1e3, r.slowness * 1e6, r.coherence, cmap="magma", vmin=0, vmax=1, shading="auto")
    s_pick, t_pick, c = r.peak(s_min=0.9 / FORMATION.vs) if name == "dipole" else r.peak(s_min=0.9 / MUD.vp)
    label = "flexural" if name == "dipole" else "Stoneley"
    ax.plot(t_pick * 1e3, s_pick * 1e6, "o", mfc="none", mec="cyan", ms=10)
    ax.text(t_pick * 1e3 + 0.05, s_pick * 1e6 + 25, f"{label} {s_pick * 1e6:.0f}", color="cyan")
    for slow, lab in ((1 / FORMATION.vp, "1/vp"), (1 / FORMATION.vs, "1/vs")):
        ax.axhline(slow * 1e6, color="w", lw=0.5, ls="--")
        ax.text(0.05, slow * 1e6 + 10, lab, color="w")
    ax.set_title(f"{name} ({run['f0'] / 1e3:.0f} kHz): slowness-time coherence")
    ax.set_xlabel("window start (ms)")
axes[0].set_ylabel("slowness (us/m)")
fig.colorbar(im, ax=axes, label="coherence")
fig.savefig(out / "fig3_dipole_stc.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ 4. flexural dispersion
run = runs["dipole"]
t, data, ref = run["t"], run["data"], run["ref"]
freqs = np.arange(1000.0, 6001.0, 100.0)
grid = np.linspace(800e-6, 1400e-6, 601)
mute = (0.25e-3 + 0.95e-3 * OFFSETS[0], 0.95e-3)  # remove the P and mud arrivals
d = post.dispersion(data, OFFSETS, t[1] - t[0], freqs, grid, mute=mute)
f_modal = np.arange(1000.0, 6001.0, 250.0)
modal = borehole_mode_slowness(f_modal, [(MUD, A)], FORMATION, order=1)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.pcolormesh(freqs / 1e3, grid * 1e6, d.semblance.T, cmap="magma", vmin=0, vmax=1, shading="auto")
ax.plot(f_modal / 1e3, modal * 1e6, "c-", lw=1.5, label="modal root-finding")
picks, _ = d.peak()
ax.plot(freqs / 1e3, picks * 1e6, "w.", ms=3, label="SEM array: semblance peak")
if ref is not None:
    picks_ref, _ = post.dispersion(ref, OFFSETS, t[1] - t[0], freqs, grid, mute=mute).peak()
    ax.plot(freqs / 1e3, picks_ref * 1e6, "o", mfc="none", mec="0.7", ms=4, markevery=5,
            label="reference array: semblance peak")
ax.axhline(1e6 / FORMATION.vs, color="w", lw=0.5, ls="--")
ax.text(1.05, 1e6 / FORMATION.vs - 25, "1/vs", color="w")
ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("phase slowness (us/m)")
ax.set_title("dipole flexural dispersion (frequency-slowness semblance)")
ax.legend(loc="upper left", fontsize=8)
fig.colorbar(im, ax=ax, label="semblance")
fig.tight_layout()
fig.savefig(out / "fig4_flexural_dispersion.png", dpi=140)
plt.close(fig)
print(f"wrote figures and results to {out}")
