"""Water over a curved seabed with a soft buried lens, on a Gmsh mesh.

    python examples/gmsh_seabed/run.py [output_dir]

Reads seabed.msh (see make_mesh.py), refines it to 5 points per wavelength
at 37.5 Hz, runs a 15 Hz monopole in the water with fluid-solid coupling,
and draws: the Gmsh mesh, the refined computational mesh, wavefield
snapshots and the hydrophone / geophone gathers (needs matplotlib).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

import insem
from insem import mesh

here = Path(__file__).parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("seabed_out")
out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
REGION_COLOR = {1: "#d8c3a5", 2: "#a9d6f5", 3: "#f2a65a"}
REGION_NAME = {1: "rock", 2: "water", 3: "lens"}

g = mesh.read_gmsh(here / "seabed.msh")
m = g.mesh


def outlines(sim):
    """Element boundary polylines from the GLL geometry: (n_elem, 4N+1, 2)."""
    x, z = sim.geometry.x, sim.geometry.z
    loop = lambda a: np.concatenate([a[:, 0, :], a[:, 1:, -1], a[:, -1, -2::-1], a[:, -2::-1, 0]], axis=1)
    return np.stack([loop(x), loop(z)], axis=-1)


def region_of_elements(sim):
    return sim.mesh.material[sim.forest.leaves[:, 0]]


# ---------------------------------------------------------------- 1. Gmsh mesh
draw = insem.Simulation(m, insem.Discretization(order=8))  # high order only to draw curved edges
draw.setup()
polys = outlines(draw)
reg = region_of_elements(draw)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
for ax, zoom in zip(axes, (None, (300, 470, -185, -95))):
    pc = PolyCollection(polys, facecolors=[REGION_COLOR[r] for r in reg], edgecolors="k", linewidths=0.35 if zoom is None else 0.7)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    if zoom is None:
        ax.set_xlim(0, 600); ax.set_ylim(-300, 100)
        ax.add_patch(Rectangle((300, -185), 170, 90, fill=False, ec="crimson", lw=1.5))
        ax.set_title(f"Gmsh mesh: {m.n_cells} quad9 macro-cells (read with insem.mesh.read_gmsh)")
        for r, c in REGION_COLOR.items():
            ax.plot([], [], "s", color=c, ms=9, mec="k", label=f"{REGION_NAME[r]} ({int((m.material == r).sum())} cells)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    else:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
        ax.set_title("zoom: curved (quadratic) cells follow the lens")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
fig.tight_layout()
fig.savefig(out / "fig1_gmsh_mesh.png", dpi=140)
plt.close(fig)

# ------------------------------------------------- simulation on the Gmsh mesh
f0 = 15.0
sim = insem.Simulation(m, insem.Discretization(order=4, f_max=2.5 * f0, points_per_wavelength=5))
sim.set_material(g.regions["rock"], insem.Material.elastic(3000.0, 1600.0, 2300.0))
sim.set_material(g.regions["water"], insem.Material.acoustic(1500.0, 1000.0))
sim.set_material(g.regions["lens"], insem.Material.elastic(1900.0, 900.0, 1950.0))
sim.set_boundary(g.boundaries["bottom"], "absorbing")
sim.set_boundary(g.boundaries["sides"], "absorbing")
sim.set_boundary(g.boundaries["sea_surface"], "free")
src = (300.0, 60.0)
sim.add_source(insem.Source.monopole(*src, insem.ricker(f0)))
xr = np.linspace(20, 580, 57)
for i, x in enumerate(xr):
    sim.add_receiver(f"h{i:02d}", float(x), 40.0, ["pressure"])
    sim.add_receiver(f"g{i:02d}", float(x), -60.0, ["velocity"], medium="elastic")
sim.set_time(duration=0.5, record_every=2, snapshot_every=1, energy_every=20)
s = sim.solver
every = max(1, int(round(0.02 / s.dt)))
sim.set_time(duration=0.5, record_every=2, snapshot_every=every, energy_every=20)
s = sim.solver
sim.save(str(out / "case.h5"))
print(f"elements {sim.dofs.n_elements}, max level {sim.forest.max_level}, dt {s.dt:.3e}, steps {s.n_steps}")
sim.run()

# ----------------------------------------------- 2. adaptive computational mesh
polys = outlines(sim)
levels = sim.forest.leaves[:, 1]
reg = region_of_elements(sim)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
cmap = plt.get_cmap("viridis", levels.max() + 1)
for ax, zoom in zip(axes, (None, (330, 430, -170, -110))):
    pc = PolyCollection(polys, facecolors=[REGION_COLOR[r] for r in reg], edgecolors=[cmap(l) for l in levels],
                        linewidths=0.4 if zoom is None else 1.2)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    if zoom is None:
        ax.set_xlim(0, 600); ax.set_ylim(-300, 100)
        ax.add_patch(Rectangle((330, -170), 100, 60, fill=False, ec="crimson", lw=1.5))
        ax.set_title(f"computational mesh: {sim.dofs.n_elements} elements after refinement to "
                     f"5 points/wavelength at {2.5 * f0:.1f} Hz")
        for l in range(levels.max() + 1):
            ax.plot([], [], "-", color=cmap(l), lw=2, label=f"level {l}: {int((levels == l).sum())} elements")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    else:
        x, z = sim.geometry.x.ravel(), sim.geometry.z.ravel()
        inside = (x > zoom[0]) & (x < zoom[1]) & (z > zoom[2]) & (z < zoom[3])
        ax.plot(x[inside], z[inside], ".", color="k", ms=1.5)
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
        ax.set_title("zoom: refined lens cells, hanging faces,\nand the GLL points")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
fig.tight_layout()
fig.savefig(out / "fig2_refined_mesh.png", dpi=140)
plt.close(fig)

# --------------------------------------------------------------- 3. snapshots
def tris(cells):
    return np.concatenate([cells[:, [0, 1, 2]], cells[:, [0, 2, 3]]])

ts = s.snapshot_times
pa = s.snapshot("acoustic", "pressure")[:, 0]
ve = s.snapshot("elastic", "velocity")[:, 1]  # vertical particle velocity
pts_a, cells_a = s.snapshot_points("acoustic"), s.snapshot_cells("acoustic")
pts_e, cells_e = s.snapshot_points("elastic"), s.snapshot_cells("elastic")
picks = [np.argmin(np.abs(ts - t)) for t in (0.12, 0.2, 0.28, 0.36)]
pmax = 0.25 * np.abs(pa[picks]).max()
vmax = 0.4 * np.abs(ve[picks]).max()
fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharex=True, sharey=True, layout="constrained")
for ax, k in zip(axes.ravel(), picks):
    ta = ax.tripcolor(pts_a[0], pts_a[1], tris(cells_a), pa[k], cmap="RdBu_r", vmin=-pmax, vmax=pmax, shading="gouraud")
    te = ax.tripcolor(pts_e[0], pts_e[1], tris(cells_e), ve[k], cmap="PuOr_r", vmin=-vmax, vmax=vmax, shading="gouraud")
    ax.plot(*src, "*", color="gold", mec="k", ms=12)
    ax.plot(xr, np.full_like(xr, 40.0), "v", color="k", ms=2)
    ax.plot(xr, np.full_like(xr, -60.0), "^", color="k", ms=2)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(380 + 90 * np.cos(th), -140 + 35 * np.sin(th), "k-", lw=0.7)
    xs = np.linspace(0, 600, 300)
    ax.plot(xs, -15 * np.sin(2 * np.pi * xs / 600) + 10 * np.cos(4 * np.pi * xs / 600), "k-", lw=0.7)
    ax.set_title(f"t = {ts[k] * 1e3:.0f} ms")
    ax.set_aspect("equal"); ax.set_xlim(0, 600); ax.set_ylim(-300, 100)
for ax in axes[1]:
    ax.set_xlabel("x (m)")
for ax in axes[:, 0]:
    ax.set_ylabel("z (m)")
fig.colorbar(ta, ax=axes[:, 1], location="right", shrink=0.8, label="pressure in water (Pa)", extend="both")
fig.colorbar(te, ax=axes[:, 1], location="right", shrink=0.8, label="vertical velocity in rock and lens (m/s)", extend="both")
fig.suptitle("15 Hz monopole in water over a curved seabed and a soft lens (fluid-solid coupling)")
fig.savefig(out / "fig3_snapshots.png", dpi=140)
plt.close(fig)

sim.write_results(str(out / "results.h5"))

# --------------------------------------------------------------- 4. gathers
t = s.times
hyd = np.array([s.trace(f"h{i:02d}", "pressure")[:, 0] for i in range(len(xr))])
geo = np.array([s.trace(f"g{i:02d}", "velocity")[:, 1] for i in range(len(xr))])
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, data, title, unit in ((axes[0], hyd, "hydrophones (z = 40 m): pressure", "Pa"),
                              (axes[1], geo, "geophones in rock (z = -60 m): vertical velocity", "m/s")):
    clip = 0.3 * np.abs(data).max()
    ax.imshow(data.T, aspect="auto", cmap="gray_r", vmin=-clip, vmax=clip,
              extent=[xr[0], xr[-1], t[-1], t[0]], interpolation="bilinear")
    scale = 0.9 * (xr[1] - xr[0]) / clip
    for x, tr in zip(xr[::4], data[::4]):
        ax.plot(x + np.clip(tr, -clip, clip) * scale, t, "k", lw=0.5)
    ax.set_title(title); ax.set_xlabel("receiver x (m)")
axes[0].set_ylabel("time (s)")
fig.tight_layout()
fig.savefig(out / "fig4_gathers.png", dpi=140)
plt.close(fig)
print(f"wrote figures, case.h5 and results.h5 to {out}")
