"""Regenerates the Gmsh fixture files used by test_gmsh.py (needs `pip install gmsh`).

    python tests/python/data/make_gmsh_fixtures.py
"""

from pathlib import Path

import gmsh

here = Path(__file__).parent


def layered(order, version, name):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("layered")
    rock = gmsh.model.occ.addRectangle(0, 0, 0, 4, 2)
    water = gmsh.model.occ.addRectangle(0, 2, 0, 4, 1.5)
    gmsh.model.occ.fragment([(2, rock)], [(2, water)])
    gmsh.model.occ.synchronize()
    eps = 1e-6
    bottom = gmsh.model.getEntitiesInBoundingBox(-eps, -eps, -eps, 4 + eps, eps, eps, 1)
    top = gmsh.model.getEntitiesInBoundingBox(-eps, 3.5 - eps, -eps, 4 + eps, 3.5 + eps, eps, 1)
    sides = [e for e in gmsh.model.getEntities(1)
             if e not in bottom and e not in top
             and abs(gmsh.model.getBoundingBox(*e)[1] - gmsh.model.getBoundingBox(*e)[4]) > eps]
    gmsh.model.addPhysicalGroup(2, [rock], 1, "rock")
    gmsh.model.addPhysicalGroup(2, [water], 2, "water")
    gmsh.model.addPhysicalGroup(1, [t for _, t in bottom], 11, "bottom")
    gmsh.model.addPhysicalGroup(1, [t for _, t in top], 12, "top")
    gmsh.model.addPhysicalGroup(1, [t for _, t in sides], 13, "sides")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # guarantees all quads
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", version)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(here / name))
    gmsh.finalize()


def annulus(name):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("annulus")
    outer = gmsh.model.occ.addDisk(0, 0, 0, 2, 2)
    inner = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    box = gmsh.model.occ.addRectangle(0, 0, 0, 3, 3)
    ring, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    quarter, _ = gmsh.model.occ.intersect(ring, [(2, box)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [t for _, t in quarter], 5, "ring")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.25)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # guarantees all quads
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(here / name))
    gmsh.finalize()


layered(1, 4.1, "layered_q4_v41.msh")
layered(2, 4.1, "layered_q9_v41.msh")
layered(1, 2.2, "layered_q4_v22.msh")
annulus("annulus_q9_v41.msh")
