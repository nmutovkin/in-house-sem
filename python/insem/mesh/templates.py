"""Structured macro-mesh templates.

Coordinates are (x, z) in Cartesian mode and (r, z) in axisymmetric mode.
Cells are counter-clockwise in that plane. Boundary faces are tagged
BOTTOM, RIGHT, TOP and LEFT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .._core import MacroMesh

# Boundary tags used by the templates.
BOTTOM = 1  # z = z_min
RIGHT = 2   # x = x_max
TOP = 3     # z = z_max
LEFT = 4    # x = x_min

MaterialSpec = Union[int, Sequence[int], Callable[[float, float], int]]
Interface = Union[float, Callable[[np.ndarray], np.ndarray]]


def _axis(spec: Union[Sequence[float], np.ndarray], n: int | None, name: str) -> np.ndarray:
    """Node coordinates along one axis: either (min, max) with n cells, or explicit breakpoints."""
    values = np.asarray(spec, dtype=float)
    if n is None:
        if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} breakpoints must be strictly increasing with at least 2 entries")
        return values
    if values.shape != (2,) or values[1] <= values[0]:
        raise ValueError(f"{name} must be (min, max) with min < max when n{name} is given")
    if n < 1:
        raise ValueError(f"n{name} must be >= 1")
    return np.linspace(values[0], values[1], n + 1)


def _lattice(breaks: np.ndarray, quadratic: bool) -> np.ndarray:
    """Node positions along one axis, with midpoints for quadratic cells."""
    if not quadratic:
        return breaks
    out = np.empty(2 * breaks.size - 1)
    out[0::2] = breaks
    out[1::2] = 0.5 * (breaks[:-1] + breaks[1:])
    return out


def _structured(xl: np.ndarray, zl: np.ndarray, quadratic: bool, material: np.ndarray, coords: str) -> MacroMesh:
    """Mesh from a node lattice: xl (mx,) column positions, zl (mz, mx) node heights per row."""
    step = 2 if quadratic else 1
    mx = xl.size
    ncx = (mx - 1) // step
    ncz = (zl.shape[0] - 1) // step
    if np.any(np.diff(zl, axis=0) <= 0):
        raise ValueError("layer interfaces must be strictly increasing in z at every x (no zero-thickness layers)")
    nodes = np.column_stack([np.broadcast_to(xl, zl.shape).ravel(), zl.ravel()])

    def node(i: int, j: int) -> int:
        return i + mx * j

    cells = []
    for j in range(ncz):
        for i in range(ncx):
            a, b = step * i, step * j
            if quadratic:
                cells.append([
                    node(a, b), node(a + 2, b), node(a + 2, b + 2), node(a, b + 2),
                    node(a + 1, b), node(a + 2, b + 1), node(a + 1, b + 2), node(a, b + 1),
                    node(a + 1, b + 1),
                ])
            else:
                cells.append([node(a, b), node(a + 1, b), node(a + 1, b + 1), node(a, b + 1)])

    boundary = []
    for j in range(ncz):
        for i in range(ncx):
            c = i + ncx * j
            if j == 0:
                boundary.append((c, 0, BOTTOM))
            if i == ncx - 1:
                boundary.append((c, 1, RIGHT))
            if j == ncz - 1:
                boundary.append((c, 2, TOP))
            if i == 0:
                boundary.append((c, 3, LEFT))
    return MacroMesh(
        "quad9" if quadratic else "quad4",
        coords,
        nodes,
        np.asarray(cells, dtype=np.int32),
        np.asarray(material, dtype=np.int32),
        np.asarray(boundary, dtype=np.int32).reshape(-1, 3),
    )


def rectangle(
    x: Sequence[float],
    z: Sequence[float],
    nx: int | None = None,
    nz: int | None = None,
    material: MaterialSpec = 0,
    quadratic: bool = False,
    coords: str = "cartesian",
) -> MacroMesh:
    """Structured mesh of a rectangle.

    Parameters
    ----------
    x, z
        Either ``(min, max)`` together with ``nx``/``nz`` cells, or explicit,
        strictly increasing node positions (``nx``/``nz`` left as None). Explicit
        positions let layer interfaces and grading be placed exactly.
    material
        A region id for all cells, a sequence with one id per cell (row by
        row, x fastest), or a function ``f(x_centre, z_centre) -> id``.
    quadratic
        Build Quad9 cells instead of Quad4.
    coords
        ``"cartesian"`` or ``"axisymmetric"`` (then x is the radius r >= 0).
    """
    xs = _axis(x, nx, "x")
    zs = _axis(z, nz, "z")
    ncx, ncz = xs.size - 1, zs.size - 1
    xl = _lattice(xs, quadratic)
    zl = np.repeat(_lattice(zs, quadratic)[:, None], xl.size, axis=1)
    if callable(material):
        xc = 0.5 * (xs[:-1] + xs[1:])
        zc = 0.5 * (zs[:-1] + zs[1:])
        mat = np.array([material(xc[i], zc[j]) for j in range(ncz) for i in range(ncx)], dtype=np.int32)
    else:
        mat = np.broadcast_to(np.asarray(material, dtype=np.int32), (ncx * ncz,)).copy()
    return _structured(xl, zl, quadratic, mat, coords)


def layered(
    x: Sequence[float],
    interfaces: Sequence[Interface],
    nz: Union[int, Sequence[int]],
    regions: Sequence[int],
    nx: int | None = None,
    quadratic: bool = False,
    coords: str = "cartesian",
) -> MacroMesh:
    """Layered model with cells that honour every interface.

    Parameters
    ----------
    x
        ``(min, max)`` with ``nx`` cells, or explicit column breakpoints.
    interfaces
        K + 1 interfaces from bottom to top, each a constant height or a
        function ``z(x)`` (vectorised over numpy arrays). Layer k lies between
        interfaces k and k + 1.
    nz
        Number of cells across each layer (one int for all layers, or K ints).
    regions
        Region id of each of the K layers.
    quadratic
        Quad9 cells, so curved interfaces are represented to second order
        (their nodes lie exactly on the interface).

    Node heights inside a layer are blended linearly between its two
    interfaces, column by column.
    """
    xs = _axis(x, nx, "x")
    k = len(interfaces) - 1
    if k < 1:
        raise ValueError("need at least two interfaces (one layer)")
    if len(regions) != k:
        raise ValueError(f"{k} layers need {k} region ids, got {len(regions)}")
    counts = [int(nz)] * k if np.isscalar(nz) else [int(n) for n in nz]
    if len(counts) != k or min(counts) < 1:
        raise ValueError(f"nz must be a positive int or {k} positive ints")

    xl = _lattice(xs, quadratic)

    def height(iface: Interface) -> np.ndarray:
        if callable(iface):
            return np.broadcast_to(np.asarray(iface(xl), dtype=float), xl.shape).astype(float)
        return np.full(xl.shape, float(iface))

    levels = [height(i) for i in interfaces]
    step = 2 if quadratic else 1
    rows = [levels[0]]
    for layer in range(k):
        n_rows = step * counts[layer]
        for r in range(1, n_rows + 1):
            t = r / n_rows
            rows.append((1.0 - t) * levels[layer] + t * levels[layer + 1])
    zl = np.vstack(rows)
    ncx = xs.size - 1
    mat = np.concatenate([np.full(ncx * counts[layer], regions[layer], dtype=np.int32) for layer in range(k)])
    return _structured(xl, zl, quadratic, mat, coords)


@dataclass
class BoreholeMesh:
    """Result of :func:`borehole`."""

    mesh: MacroMesh
    #: region name -> region id
    regions: Dict[str, int]
    #: boundary name -> tag ("axis" takes no boundary condition)
    boundaries: Dict[str, int]
    #: radial and vertical cell breakpoints
    r: np.ndarray
    z: np.ndarray


def _graded(r0: float, r1: float, first: float, growth: float, max_size: float) -> List[float]:
    """Breakpoints from r0 to r1: cells start at `first`, grow by `growth` up to `max_size`."""
    out, r, size = [r0], r0, first
    while r1 - r > 1e-12 * max(1.0, abs(r1)):
        size = min(size, max_size)
        if r1 - r <= 1.5 * size:  # merge a short remainder into the last cell
            out.append(r1)
            break
        r += size
        out.append(r)
        size *= growth
    return out


def borehole(
    z: Sequence[float],
    annuli: Sequence[Tuple[str, float]],
    formation_radius: float,
    beds: Sequence[Tuple[float, str]] = (),
    dz: float = 0.05,
    dr: Optional[float] = None,
    growth: float = 1.3,
    fluid_cells: int = 2,
    quadratic: bool = False,
) -> BoreholeMesh:
    """Axisymmetric (r, z) mesh of a borehole: concentric annuli around the
    axis, then a radially graded formation split into horizontal beds.

    Parameters
    ----------
    z
        (z_min, z_max) of the model.
    annuli
        (region name, outer radius) from the axis outward, e.g.
        ``[("mud", 0.1), ("casing", 0.11), ("cement", 0.15)]``. Each spans the
        full height. The first one (the fluid column) gets ``fluid_cells``
        radial cells; the others ceil(thickness / dz) cells.
    formation_radius
        Outer radius of the model.
    beds
        (z_bottom, region name) of the formation beds, bottom to top; the first
        z_bottom must be z_min. Default: one bed named "formation".
    dz
        Target vertical cell size (bed boundaries are honoured exactly).
    dr, growth
        Formation cells start at the size of the last annulus cell (at most
        dr) and grow by `growth` per cell up to dr (default dr = 4 dz).

    Region ids are 1, 2, ... in order of first appearance (annuli, then
    beds). Boundary tags: "bottom" (1), "outer" (2), "top" (3) and "axis"
    (4, which takes no boundary condition). Wavelength-based refinement in
    `Discretization` can refine any of these further.
    """
    z0, z1 = float(z[0]), float(z[1])
    if not annuli:
        raise ValueError("borehole needs at least the fluid column in `annuli`")
    radii = [float(r) for _, r in annuli]
    if any(b <= a for a, b in zip([0.0] + radii, radii)) or formation_radius <= radii[-1]:
        raise ValueError("annulus radii must increase and stay below formation_radius")
    beds = list(beds) or [(z0, "formation")]
    if abs(beds[0][0] - z0) > 1e-12 * max(1.0, abs(z0)) or any(b[0] >= c[0] for b, c in zip(beds, beds[1:])):
        raise ValueError("beds must start at z_min and be sorted by increasing z_bottom")
    dr = 4 * dz if dr is None else dr

    # Radial breakpoints.
    r_breaks = list(np.linspace(0.0, radii[0], fluid_cells + 1))
    last = radii[0] / fluid_cells
    for inner, outer in zip(radii, radii[1:]):
        n = max(1, int(np.ceil((outer - inner) / dz - 1e-9)))
        r_breaks += list(np.linspace(inner, outer, n + 1)[1:])
        last = (outer - inner) / n
    r_breaks += _graded(radii[-1], formation_radius, min(last * growth, dr), growth, dr)[1:]

    # Vertical breakpoints honouring the bed boundaries.
    tops = [b[0] for b in beds[1:]] + [z1]
    z_breaks = [z0]
    for bottom, top in zip([b[0] for b in beds], tops):
        n = max(1, int(np.ceil((top - bottom) / dz - 1e-9)))
        z_breaks += list(np.linspace(bottom, top, n + 1)[1:])

    names: Dict[str, int] = {}
    for name in [a for a, _ in annuli] + [b for _, b in beds]:
        names.setdefault(name, len(names) + 1)
    bed_bottoms = np.array([b[0] for b in beds])

    def region(rc: float, zc: float) -> int:
        for name, outer in annuli:
            if rc < outer:
                return names[name]
        return names[beds[int(np.searchsorted(bed_bottoms, zc, side="right")) - 1][1]]

    m = rectangle(r_breaks, z_breaks, material=region, quadratic=quadratic, coords="axisymmetric")
    return BoreholeMesh(m, names, {"bottom": BOTTOM, "outer": RIGHT, "top": TOP, "axis": LEFT},
                        np.asarray(r_breaks), np.asarray(z_breaks))
