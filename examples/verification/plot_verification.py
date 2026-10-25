"""Verification plots: SEM against independent reference solutions.

    python examples/verification/plot_verification.py [output.png]

Lamb's problem (vertical surface line load) against a wavenumber-integration
reference, and a double couple against the analytic 2D Green's function.
The references live in tests/python/analytic.py (needs scipy, matplotlib).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import insem
from insem import mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "python"))
import analytic  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("verification.png")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
# (a) Lamb's problem
vp, vs, rho = 2.0, 1.0, 1.0
for order, style in ((4, "C1--"), (6, "C0-")):
    lm = insem.Simulation(mesh.rectangle((-12, 12), (-10, 0), nx=96, nz=40), insem.Discretization(order=order))
    lm.set_material(0, insem.Material.elastic(vp, vs, rho))
    for tag in (mesh.BOTTOM, mesh.LEFT, mesh.RIGHT):
        lm.set_boundary(tag, "absorbing")
    stf = insem.ricker(1.0)
    lm.add_source(insem.Source.force(0.0, 0.0, 0.0, -1.0, stf))
    lm.add_receiver("r", 4.0, 0.0, ["displacement"])
    lm.set_time(duration=6.5, record_every=2)
    lm.run()
    ls = lm.solver
    u = ls.trace("r", "displacement")
    if order == 4:
        ref = analytic.lamb_surface(ls.times, stf(ls.times), 4.0, vp, vs, rho)
        axes[0].plot(ls.times, ref[:, 1], "k-", lw=2.5, alpha=0.35, label="reference (wavenumber integration)")
    err = np.linalg.norm(u - analytic.lamb_surface(ls.times, stf(ls.times), 4.0, vp, vs, rho)) / np.linalg.norm(ref)
    axes[0].plot(ls.times, u[:, 1], style, lw=1, label=f"SEM order {order} ({100 * err:.1f} % error)")
axes[0].set_title("Lamb's problem: u_z at x = 4 on the surface")
axes[0].set_xlabel("time"); axes[0].legend(fontsize=7, loc="lower left")
axes[0].annotate("P", (2.5, 0.0)); axes[0].annotate("Rayleigh", (4.6, -0.115))
# (b) elastic Green's function, double couple
gs = insem.Simulation(mesh.rectangle((-5, 5), (-5, 5), nx=48, nz=48), insem.Discretization(order=5))
gs.set_material(0, insem.Material.elastic(2.0, 1.15, 1.0))
for tag in (1, 2, 3, 4):
    gs.set_boundary(tag, "absorbing")
stf = insem.ricker(1.0)
gs.add_source(insem.Source.moment_tensor(0, 0, 0.0, 0.0, 1.0, stf))
gs.add_receiver("r", 1.9, 1.1, ["displacement"])
gs.set_time(duration=4.0)
gs.run()
gss = gs.solver
ua = analytic.elastic_displacement(gss.times, stf(gss.times), 1.9, 1.1, 2.0, 1.15, 1.0, moment=((0, 1), (1, 0)))
for c, name in enumerate("xz"):
    axes[1].plot(gss.times, ua[:, c], "k-", lw=2.5, alpha=0.35, label="analytic" if c == 0 else None)
    axes[1].plot(gss.times, gss.trace("r", "displacement")[:, c], f"C{c}-", lw=1, label=f"SEM u_{name}")
axes[1].set_title(f"double couple, 2D Green's function ({100 * analytic.rel_l2(gss.trace('r', 'displacement'), ua):.2f} % error)")
axes[1].set_xlabel("time"); axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=140)
print(f"wrote {out}")
