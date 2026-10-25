"""Reader for Gmsh ``.msh`` files (format 2.2 and 4.1, ASCII).

Only quadrilateral cells are used (SEM needs quads). Gmsh produces an
all-quad mesh with ``Mesh.RecombineAll = 1`` together with
``Mesh.SubdivisionAlgorithm = 1``; recombination alone may leave triangles.

* 4-node quadrangles (Gmsh type 3) -> ``quad4`` cells
* 9-node quadrangles (Gmsh type 10) -> ``quad9`` cells (curved)

Surface elements give material regions: the physical tag of their surface,
or the geometric entity tag when the file has no physical groups. Line
elements on the domain boundary (Gmsh types 1 and 8) give boundary tags the
same way; lines inside the domain (e.g. material interfaces) are ignored,
since interfaces follow from the regions. Cells are reordered to be
counter-clockwise. The mesh plane is the Gmsh x-y plane by default; its
coordinates become (x, z), or (r, z) for axisymmetric models.

The Gmsh program and its Python package are not needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .._core import MacroMesh

_QUAD4, _QUAD9, _LINE2, _LINE3 = 3, 10, 1, 8
_TRIANGLES = {2, 9, 20, 21, 22, 23, 24, 25}
_NODES_PER_TYPE = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5, 8: 3, 9: 6, 10: 9, 11: 10, 15: 1, 16: 8, 21: 10}


@dataclass
class GmshMesh:
    """Result of :func:`read_gmsh`."""

    mesh: MacroMesh
    #: physical surface names -> region ids
    regions: Dict[str, int] = field(default_factory=dict)
    #: physical curve names -> boundary tags
    boundaries: Dict[str, int] = field(default_factory=dict)
    #: number of tagged line elements inside the domain (ignored)
    interior_lines: int = 0


def _sections(path: Path) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    name = None
    body: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("$"):
                if line.startswith("$End"):
                    sections[name] = body
                    name, body = None, []
                else:
                    name, body = line[1:], []
            elif name is not None:
                body.append(line)
    return sections


def _physical_names(lines: List[str]) -> Dict[Tuple[int, int], str]:
    names = {}
    for line in lines[1:]:
        dim, tag, rest = line.split(maxsplit=2)
        names[(int(dim), int(tag))] = rest.strip().strip('"')
    return names


def _parse_v2(sec):
    lines = sec["Nodes"]
    nodes = {}
    for line in lines[1:]:
        p = line.split()
        nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    elements = []  # (type, physical or entity tag, node tags)
    for line in sec["Elements"][1:]:
        p = [int(v) for v in line.split()]
        etype, ntags = p[1], p[2]
        tags = p[3:3 + ntags]
        tag = tags[0] if ntags >= 1 and tags[0] != 0 else (tags[1] if ntags >= 2 else 0)
        elements.append((etype, tag, p[3 + ntags:]))
    return nodes, elements


def _parse_v4(sec):
    # Entity -> physical tag (first one), per dimension.
    physical = {}
    if "Entities" in sec:
        tokens = " ".join(sec["Entities"]).split()
        pos = 0

        def take(n=1):
            nonlocal pos
            vals = tokens[pos:pos + n]
            pos += n
            return vals

        counts = [int(v) for v in take(4)]
        for dim, count in enumerate(counts):
            for _ in range(count):
                tag = int(take()[0])
                take(3 if dim == 0 else 6)  # point coordinates or bounding box
                nphys = int(take()[0])
                phys = [int(v) for v in take(nphys)]
                if dim > 0:
                    nbound = int(take()[0])
                    take(nbound)
                physical[(dim, tag)] = phys[0] if phys else None

    tokens = " ".join(sec["Nodes"]).split()
    pos = 0
    nblocks = int(tokens[0])
    pos = 4
    nodes = {}
    for _ in range(nblocks):
        dim, ent, parametric, n = (int(v) for v in tokens[pos:pos + 4])
        pos += 4
        ids = [int(v) for v in tokens[pos:pos + n]]
        pos += n
        width = 3 + (dim if parametric else 0)
        for i, nid in enumerate(ids):
            c = tokens[pos + i * width:pos + i * width + 3]
            nodes[nid] = (float(c[0]), float(c[1]), float(c[2]))
        pos += n * width

    tokens = " ".join(sec["Elements"]).split()
    nblocks = int(tokens[0])
    pos = 4
    elements = []
    for _ in range(nblocks):
        dim, ent, etype, n = (int(v) for v in tokens[pos:pos + 4])
        pos += 4
        npe = _NODES_PER_TYPE.get(etype)
        if npe is None:
            raise ValueError(f"unsupported Gmsh element type {etype}")
        phys = physical.get((dim, ent))
        tag = phys if phys is not None else ent
        for _ in range(n):
            vals = [int(v) for v in tokens[pos:pos + 1 + npe]]
            pos += 1 + npe
            elements.append((etype, tag, vals[1:]))
    return nodes, elements


def read_gmsh(path, plane: str = "xy", coords: str = "cartesian") -> GmshMesh:
    """Read a Gmsh mesh of quadrilaterals.

    Parameters
    ----------
    path
        ``.msh`` file, format 2.2 or 4.1, ASCII.
    plane
        Which Gmsh coordinates are the model plane: ``"xy"`` (default) or ``"xz"``.
    coords
        ``"cartesian"`` or ``"axisymmetric"`` (then the first coordinate is r >= 0).
    """
    path = Path(path)
    sec = _sections(path)
    if "MeshFormat" not in sec:
        raise ValueError(f"{path} is not a Gmsh mesh file")
    version, file_type = sec["MeshFormat"][0].split()[:2]
    if file_type != "0":
        raise ValueError(f"{path}: binary Gmsh files are not supported; save as ASCII")
    if version.startswith("2"):
        nodes, elements = _parse_v2(sec)
    elif version.startswith("4"):
        nodes, elements = _parse_v4(sec)
    else:
        raise ValueError(f"{path}: unsupported Gmsh format {version} (use 2.2 or 4.1)")
    names = _physical_names(sec["PhysicalNames"]) if "PhysicalNames" in sec else {}

    if any(etype in _TRIANGLES for etype, _, _ in elements):
        raise ValueError(f"{path} contains triangles; SEM needs quadrilaterals "
                         "(mesh with 'Mesh.RecombineAll = 1' and 'Mesh.SubdivisionAlgorithm = 1')")
    quads = [(etype, tag, conn) for etype, tag, conn in elements if etype in (_QUAD4, _QUAD9)]
    if not quads:
        raise ValueError(f"{path} contains no quadrilateral cells")
    types = {etype for etype, _, _ in quads}
    if len(types) > 1:
        raise ValueError(f"{path} mixes 4-node and 9-node quadrangles")
    quadratic = types == {_QUAD9}

    # Compact node numbering over the nodes cells use.
    used = sorted({n for _, _, conn in quads for n in conn})
    index = {n: i for i, n in enumerate(used)}
    axes = {"xy": (0, 1), "xz": (0, 2)}[plane]
    xyz = np.array([nodes[n] for n in used])
    points = xyz[:, list(axes)]

    cells, material = [], []
    for _, tag, conn in quads:
        c = [index[n] for n in conn]
        # Signed area of the corner polygon; reverse clockwise cells.
        p = points[c[:4]]
        area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        if area < 0:
            c = [c[0], c[3], c[2], c[1]] + ([c[7], c[6], c[5], c[4], c[8]] if quadratic else [])
        cells.append(c)
        material.append(tag)
    cells = np.asarray(cells, dtype=np.int32)

    # Boundary tags from line elements that coincide with an unshared cell face.
    faces = {}
    for ci, c in enumerate(cells):
        for f in range(4):
            key = tuple(sorted((c[f], c[(f + 1) % 4])))
            faces.setdefault(key, []).append((ci, f))
    boundary, interior = [], 0
    for etype, tag, conn in elements:
        if etype not in (_LINE2, _LINE3):
            continue
        if conn[0] not in index or conn[1] not in index:
            continue
        key = tuple(sorted((index[conn[0]], index[conn[1]])))
        owners = faces.get(key, [])
        if len(owners) == 1:
            boundary.append((owners[0][0], owners[0][1], tag))
        elif len(owners) == 2:
            interior += 1
    boundary = np.asarray(boundary, dtype=np.int32).reshape(-1, 3)

    macro = MacroMesh("quad9" if quadratic else "quad4", coords, points, cells,
                      np.asarray(material, dtype=np.int32), boundary)
    return GmshMesh(
        mesh=macro,
        regions={name: tag for (dim, tag), name in names.items() if dim == 2},
        boundaries={name: tag for (dim, tag), name in names.items() if dim == 1},
        interior_lines=interior,
    )
