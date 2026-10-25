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
    Material,
    Receiver,
    Simulation,
    Solver,
    Source,
    SourceTimeFunction,
    TimeConfig,
    __version__,
    gaussian,
    gaussian_derivative,
    global_coordinates,
    real_dtype,
    ricker,
    sampled_stf,
)
from . import basis, mesh, post

__all__ = [
    "Discretization",
    "DofMap",
    "Forest",
    "Geometry",
    "InsemError",
    "MacroMesh",
    "Material",
    "Receiver",
    "Simulation",
    "Solver",
    "Source",
    "SourceTimeFunction",
    "TimeConfig",
    "__version__",
    "basis",
    "gaussian",
    "gaussian_derivative",
    "global_coordinates",
    "mesh",
    "post",
    "real_dtype",
    "ricker",
    "sampled_stf",
]
