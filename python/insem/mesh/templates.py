"""Structured macro-mesh templates.

Coordinates are (x, z) in Cartesian mode and (r, z) in axisymmetric mode.
Cells are counter-clockwise in that plane.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np

from .._core import MacroMesh

# Boundary tags used by the templates.
BOTTOM = 1  # z = z_min
RIGHT = 2   # x = x_max
TOP = 3     # z = z_max
LEFT = 4    # x = x_min

MaterialSpec = Union[int, Sequence[int], Callable[[float, float], int]]


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

    Boundary faces are tagged BOTTOM, RIGHT, TOP and LEFT.
    """
    xs = _axis(x, nx, "x")
    zs = _axis(z, nz, "z")
    ncx, ncz = xs.size - 1, zs.size - 1

    if quadratic:
        # Node lattice with midpoints, so each cell spans 3 x 3 lattice nodes.
        xl = np.empty(2 * ncx + 1)
        xl[0::2] = xs
        xl[1::2] = 0.5 * (xs[:-1] + xs[1:])
        zl = np.empty(2 * ncz + 1)
        zl[0::2] = zs
        zl[1::2] = 0.5 * (zs[:-1] + zs[1:])
    else:
        xl, zl = xs, zs
    mx = xl.size
    xx, zz = np.meshgrid(xl, zl)  # shape (len(zl), len(xl)), x fastest
    nodes = np.column_stack([xx.ravel(), zz.ravel()])

    def node(i: int, j: int) -> int:
        return i + mx * j

    cells = []
    for j in range(ncz):
        for i in range(ncx):
            if quadratic:
                a, b = 2 * i, 2 * j
                cells.append([
                    node(a, b), node(a + 2, b), node(a + 2, b + 2), node(a, b + 2),
                    node(a + 1, b), node(a + 2, b + 1), node(a + 1, b + 2), node(a, b + 1),
                    node(a + 1, b + 1),
                ])
            else:
                cells.append([node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)])
    cells = np.asarray(cells, dtype=np.int32)

    ncell = ncx * ncz
    if callable(material):
        xc = 0.5 * (xs[:-1] + xs[1:])
        zc = 0.5 * (zs[:-1] + zs[1:])
        mat = np.array([material(xc[i], zc[j]) for j in range(ncz) for i in range(ncx)], dtype=np.int32)
    else:
        mat = np.broadcast_to(np.asarray(material, dtype=np.int32), (ncell,)).copy()

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
    boundary = np.asarray(boundary, dtype=np.int32).reshape(-1, 3)

    return MacroMesh("quad9" if quadratic else "quad4", coords, nodes, cells, mat, boundary)
