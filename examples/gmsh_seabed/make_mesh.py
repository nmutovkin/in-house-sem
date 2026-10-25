"""Gmsh model of a curved seabed over rock with a buried elliptical lens.

    pip install gmsh
    python examples/gmsh_seabed/make_mesh.py [examples/gmsh_seabed/seabed.msh]

All-quad (recombined + subdivided), second-order (quad9) cells, so the
seabed and the lens are curved. Physical groups: surfaces rock (1),
water (2), lens (3); curves bottom (11), sea_surface (12), sides (13).
Curvature-based sizing is off on purpose: it puts ~3 m cells at the ends
of the ellipse, and the smallest cell sets the global time step; insem's
wavelength-based refinement adds resolution where the waves need it.
The generated seabed.msh is committed, so running the example does not
need Gmsh.
"""
import sys
from pathlib import Path

import gmsh
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("seabed.msh"))
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("seabed")
occ = gmsh.model.occ
W, B, T = 600.0, -300.0, 100.0
xs = np.linspace(0, W, 13)
zs = -15 * np.sin(2 * np.pi * xs / W) + 10 * np.cos(4 * np.pi * xs / W)  # seabed z(x)
pts = [occ.addPoint(x, z, 0) for x, z in zip(xs, zs)]
seabed = occ.addSpline(pts)
p_bl, p_br = occ.addPoint(0, B, 0), occ.addPoint(W, B, 0)
p_tl, p_tr = occ.addPoint(0, T, 0), occ.addPoint(W, T, 0)
rock_loop = occ.addCurveLoop([occ.addLine(p_bl, p_br), occ.addLine(p_br, pts[-1]), -seabed, occ.addLine(pts[0], p_bl)])
water_loop = occ.addCurveLoop([seabed, occ.addLine(pts[-1], p_tr), occ.addLine(p_tr, p_tl), occ.addLine(p_tl, pts[0])])
rock = occ.addPlaneSurface([rock_loop])
water = occ.addPlaneSurface([water_loop])
lens = occ.addDisk(380, -140, 0, 90, 35)
occ.fragment([(2, rock), (2, water)], [(2, lens)])
occ.synchronize()

groups = {"rock": [], "water": [], "lens": []}
for dim, tag in gmsh.model.getEntities(2):
    x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(dim, tag)
    if x0 > 280 and x1 < 480 and y0 > -180 and y1 < -100:
        groups["lens"].append(tag)
    elif y0 > -40:
        groups["water"].append(tag)
    else:
        groups["rock"].append(tag)
for i, (name, tags) in enumerate(groups.items(), start=1):
    gmsh.model.addPhysicalGroup(2, tags, i, name)
eps = 1e-3
def curves(x0, y0, x1, y1):
    return [t for _, t in gmsh.model.getEntitiesInBoundingBox(x0 - eps, y0 - eps, -eps, x1 + eps, y1 + eps, eps, 1)]
gmsh.model.addPhysicalGroup(1, curves(0, B, W, B), 11, "bottom")
gmsh.model.addPhysicalGroup(1, curves(0, T, W, T), 12, "sea_surface")
gmsh.model.addPhysicalGroup(1, curves(0, B, 0, T) + curves(W, B, W, T), 13, "sides")
gmsh.option.setNumber("Mesh.MeshSizeMax", 50)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.model.mesh.generate(2)
gmsh.write(out)
gmsh.finalize()
