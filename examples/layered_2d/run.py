"""Two-layer elastic model with a curved interface, a moment-tensor source and
a vertical receiver line (a 2D stand-in for a downhole array).

    python examples/layered_2d/run.py [output_dir]

Writes case.h5 (re-runnable with `insem-run case.h5`) and results.h5, and a
plot of the gather if matplotlib is installed.
"""

import sys
from pathlib import Path

import numpy as np

import insem
from insem import mesh

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)

# Geometry: x in [0, 600] m, z in [-400, 0] m, interface z = -180 + 20 sin(x / 80).
model = mesh.layered(
    (0.0, 600.0),
    [-400.0, lambda x: -180.0 + 20.0 * np.sin(x / 80.0), 0.0],
    nz=[14, 12],
    regions=[1, 2],
    nx=40,
    quadratic=True,
)
sim = insem.Simulation(model, insem.Discretization(order=4))
sim.set_material(1, insem.Material.elastic(vp=3500.0, vs=2000.0, rho=2500.0))  # lower layer
sim.set_material(2, insem.Material.elastic(vp=2500.0, vs=1300.0, rho=2200.0))  # upper layer
for tag in (mesh.BOTTOM, mesh.LEFT, mesh.RIGHT):
    sim.set_boundary(tag, "absorbing")
sim.set_boundary(mesh.TOP, "free")

# Microseismic-like double couple below the interface, 30 Hz Ricker.
sim.add_source(insem.Source.moment_tensor(380.0, -260.0, mxx=0.0, mzz=0.0, mxz=1e9, stf=insem.ricker(30.0)))
depths = np.linspace(-20.0, -360.0, 18)
for i, z in enumerate(depths):
    sim.add_receiver(f"g{i:02d}", 150.0, z, ["velocity"])
sim.set_time(duration=0.35, energy_every=50)

sim.save(str(out / "case.h5"))
sim.run()
sim.write_results(str(out / "results.h5"))

s = sim.solver
print(f"elements: {sim.dofs.n_elements}, dt = {s.dt:.3e} s ({s.n_steps} steps), "
      f"stable limit {s.stable_dt:.3e} s")
print(f"wrote {out / 'case.h5'} and {out / 'results.h5'}")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

t = s.times
gather = np.array([s.trace(f"g{i:02d}", "velocity")[:, 1] for i in range(len(depths))])
scale = 0.5 * np.abs(depths[1] - depths[0]) / np.abs(gather).max()
fig, ax = plt.subplots(figsize=(6, 6))
for z, tr in zip(depths, gather):
    ax.plot(t, z + scale * tr, "k", lw=0.8)
ax.set_xlabel("time (s)")
ax.set_ylabel("receiver depth z (m)")
ax.set_title("vertical particle velocity")
fig.tight_layout()
fig.savefig(out / "gather.png", dpi=120)
print(f"wrote {out / 'gather.png'}")
