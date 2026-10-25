"""insem: spectral element simulator for coupled acoustic-elastic wave propagation.

The numerical core is C++ (``insem._core``); this package re-exports its object
model and adds Python-side tools such as mesh templates.
"""

from ._core import (
    Discretization,
    DofMap,
    Forest,
    Geometry,
    InsemError,
    MacroMesh,
    Simulation,
    __version__,
    global_coordinates,
    real_dtype,
)
from . import basis, mesh

__all__ = [
    "Discretization",
    "DofMap",
    "Forest",
    "Geometry",
    "InsemError",
    "MacroMesh",
    "Simulation",
    "__version__",
    "basis",
    "global_coordinates",
    "mesh",
    "real_dtype",
]
