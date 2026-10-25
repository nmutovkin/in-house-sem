"""YAML front-end: a thin layer that builds a :class:`insem.Simulation`.

A configuration maps one-to-one onto the Python API (docs/DESIGN.md §7.1).
Unknown keys are errors. Relative file paths are relative to the YAML file.

Example::

    mesh:
      layered: {x: [0, 600], nx: 40, interfaces: [-400, "-180 + 20*sin(x/80)", 0],
                nz: [14, 12], regions: [1, 2], quadratic: true}
    discretization: {order: 4}
    materials:
      1: {medium: elastic, vp: 3500, vs: 2000, rho: 2500}
      2: {medium: elastic, vp: 2500, vs: 1300, rho: 2200}
    boundaries: {bottom: absorbing, left: absorbing, right: absorbing, top: free}
    sources:
      - {type: moment_tensor, position: [380, -260], mxz: 1.0e9, stf: {type: ricker, f0: 30}}
    receivers:
      - {line: {start: [150, -20], end: [150, -360], n: 18}, prefix: g, quantities: [velocity]}
    time: {duration: 0.35}
    output: {segy: [{quantity: velocity, component: z}], xdmf: false}

Mesh: exactly one of ``rectangle`` (arguments of ``mesh.rectangle``),
``layered`` (``mesh.layered``; an interface may be a formula in x) or
``gmsh`` (``{file: ..., plane: xy}``), plus optional ``coords``.
Materials and boundaries are keyed by region / tag ids or by names: the
template boundary names bottom/right/top/left, or Gmsh physical names.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import _core, mesh as _mesh

_TEMPLATE_TAGS = {"bottom": _mesh.BOTTOM, "right": _mesh.RIGHT, "top": _mesh.TOP, "left": _mesh.LEFT}


class ConfigError(ValueError):
    """An invalid configuration."""


def _check_keys(section: str, d: Dict[str, Any], allowed, required=()) -> None:
    if not isinstance(d, dict):
        raise ConfigError(f"{section}: expected a mapping, got {type(d).__name__}")
    unknown = set(d) - set(allowed)
    if unknown:
        raise ConfigError(f"{section}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    missing = [k for k in required if k not in d]
    if missing:
        raise ConfigError(f"{section}: missing key(s) {missing}")


# ------------------------------------------------------------ expressions

_FUNCS = {name: getattr(np, name) for name in
          ("sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "abs", "arctan", "minimum", "maximum")}
_CONSTS = {"pi": math.pi, "e": math.e}


def _formula(text: str):
    """A vectorised function of x from a formula; only arithmetic, x, pi, e
    and a few numpy functions are allowed (no attribute access, no imports)."""
    tree = ast.parse(text, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
                             ast.operator, ast.unaryop)):
            continue
        if isinstance(node, ast.Name) and (node.id == "x" or node.id in _FUNCS or node.id in _CONSTS):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            continue
        raise ConfigError(f"formula '{text}': '{ast.unparse(node)}' is not allowed")
    code = compile(tree, "<formula>", "eval")
    return lambda x: eval(code, {"__builtins__": {}}, {**_FUNCS, **_CONSTS, "x": np.asarray(x)})


# ------------------------------------------------------------------- build


@dataclass
class Built:
    """A simulation built from a configuration, with its output settings."""

    simulation: _core.Simulation
    output: Dict[str, Any] = field(default_factory=dict)
    base_dir: Path = Path(".")
    names: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _build_mesh(cfg, base: Path):
    _check_keys("mesh", cfg, {"rectangle", "layered", "gmsh", "coords"})
    kinds = [k for k in ("rectangle", "layered", "gmsh") if k in cfg]
    if len(kinds) != 1:
        raise ConfigError("mesh: give exactly one of 'rectangle', 'layered' or 'gmsh'")
    coords = cfg.get("coords", "cartesian")
    kind = kinds[0]
    args = dict(cfg[kind])
    names = {"regions": {}, "boundaries": dict(_TEMPLATE_TAGS)}
    if kind == "rectangle":
        _check_keys("mesh.rectangle", args, {"x", "z", "nx", "nz", "material", "quadratic"}, ["x", "z"])
        m = _mesh.rectangle(coords=coords, **args)
    elif kind == "layered":
        _check_keys("mesh.layered", args, {"x", "nx", "interfaces", "nz", "regions", "quadratic"},
                    ["x", "interfaces", "nz", "regions"])
        args["interfaces"] = [_formula(i) if isinstance(i, str) else float(i) for i in args["interfaces"]]
        m = _mesh.layered(coords=coords, **args)
    else:
        _check_keys("mesh.gmsh", args, {"file", "plane"}, ["file"])
        g = _mesh.read_gmsh(base / args["file"], plane=args.get("plane", "xy"), coords=coords)
        m = g.mesh
        names = {"regions": g.regions, "boundaries": g.boundaries}
    return m, names


def _resolve(key, table: Dict[str, int], what: str) -> int:
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.lstrip("-").isdigit():
        return int(key)
    if key in table:
        return table[key]
    raise ConfigError(f"unknown {what} '{key}' (known names: {sorted(table)})")


def _stf(cfg, base: Path):
    _check_keys("stf", cfg, {"type", "f0", "t0", "amplitude", "values", "file", "dt"}, ["type"])
    kind = cfg["type"]
    amplitude = float(cfg.get("amplitude", 1.0))
    if kind in ("ricker", "gaussian", "gaussian_derivative"):
        if "f0" not in cfg:
            raise ConfigError(f"stf: '{kind}' needs f0")
        fn = {"ricker": _core.ricker, "gaussian": _core.gaussian, "gaussian_derivative": _core.gaussian_derivative}[kind]
        return fn(float(cfg["f0"]), cfg.get("t0"), amplitude)
    if kind == "samples":
        if "dt" not in cfg or ("values" in cfg) == ("file" in cfg):
            raise ConfigError("stf: 'samples' needs dt and exactly one of values / file")
        values = np.asarray(cfg["values"], float) if "values" in cfg else np.loadtxt(base / cfg["file"])
        return _core.sampled_stf(values, float(cfg["dt"]), amplitude)
    raise ConfigError(f"stf: unknown type '{kind}' (ricker, gaussian, gaussian_derivative, samples)")


def _source(cfg, base: Path):
    _check_keys("source", cfg, {"type", "position", "fx", "fz", "mxx", "mzz", "mxz", "stf"},
                ["type", "position", "stf"])
    x, z = (float(v) for v in cfg["position"])
    stf = _stf(cfg["stf"], base)
    kind = cfg["type"]
    if kind == "force":
        return _core.Source.force(x, z, float(cfg.get("fx", 0.0)), float(cfg.get("fz", 0.0)), stf)
    if kind == "moment_tensor":
        return _core.Source.moment_tensor(x, z, float(cfg.get("mxx", 0.0)), float(cfg.get("mzz", 0.0)),
                                          float(cfg.get("mxz", 0.0)), stf)
    if kind == "monopole":
        return _core.Source.monopole(x, z, stf)
    raise ConfigError(f"source: unknown type '{kind}' (force, moment_tensor, monopole)")


def _receivers(cfg) -> List[tuple]:
    if "line" in cfg:
        _check_keys("receiver line", cfg, {"line", "prefix", "quantities", "medium"}, ["line", "quantities"])
        line = cfg["line"]
        _check_keys("receiver line.line", line, {"start", "end", "n"}, ["start", "end", "n"])
        n = int(line["n"])
        prefix = cfg.get("prefix", "r")
        start, end = np.asarray(line["start"], float), np.asarray(line["end"], float)
        points = [start + (end - start) * (i / (n - 1) if n > 1 else 0.0) for i in range(n)]
        width = len(str(n - 1))
        return [(f"{prefix}{i:0{width}d}", float(p[0]), float(p[1]), cfg["quantities"], cfg.get("medium", ""))
                for i, p in enumerate(points)]
    _check_keys("receiver", cfg, {"name", "position", "quantities", "medium"}, ["name", "position", "quantities"])
    x, z = (float(v) for v in cfg["position"])
    return [(str(cfg["name"]), x, z, cfg["quantities"], cfg.get("medium", ""))]


def build(cfg: Dict[str, Any], base_dir=".") -> Built:
    """Build a simulation from a configuration mapping."""
    base = Path(base_dir)
    _check_keys("config", cfg, {"mesh", "discretization", "materials", "boundaries", "sources", "receivers",
                                "time", "output"}, ["mesh", "materials", "time"])
    m, names = _build_mesh(cfg["mesh"], base)

    disc_cfg = dict(cfg.get("discretization", {}))
    _check_keys("discretization", disc_cfg, {"order", "uniform_level", "f_max", "points_per_wavelength",
                                             "max_level", "regions"})
    regions = []
    for r in disc_cfg.pop("regions", []):
        _check_keys("discretization.regions", r, {"x", "z", "level"}, ["x", "z", "level"])
        regions.append((float(r["x"][0]), float(r["x"][1]), float(r["z"][0]), float(r["z"][1]), int(r["level"])))
    sim = _core.Simulation(m, _core.Discretization(regions=regions, **disc_cfg))

    for key, mat in cfg["materials"].items():
        _check_keys(f"materials.{key}", mat, {"medium", "vp", "vs", "rho"}, ["medium", "vp", "rho"])
        region = _resolve(key, names["regions"], "region")
        if mat["medium"] == "acoustic":
            sim.set_material(region, _core.Material.acoustic(float(mat["vp"]), float(mat["rho"])))
        elif mat["medium"] == "elastic":
            if "vs" not in mat:
                raise ConfigError(f"materials.{key}: elastic material needs vs")
            sim.set_material(region, _core.Material.elastic(float(mat["vp"]), float(mat["vs"]), float(mat["rho"])))
        else:
            raise ConfigError(f"materials.{key}: medium must be 'acoustic' or 'elastic'")

    for key, kind in cfg.get("boundaries", {}).items():
        sim.set_boundary(_resolve(key, names["boundaries"], "boundary"), str(kind))
    for src in cfg.get("sources", []):
        sim.add_source(_source(src, base))
    for rec in cfg.get("receivers", []):
        for name, x, z, quantities, medium in _receivers(rec):
            sim.add_receiver(name, x, z, list(quantities), medium)

    t = dict(cfg["time"])
    _check_keys("time", t, {"duration", "n_steps", "dt", "safety", "record_every", "energy_every",
                            "snapshot_every", "snapshot_quantities"})
    if "duration" not in t and "n_steps" not in t:
        raise ConfigError("time: give duration or n_steps")
    sim.set_time(**t)

    output = dict(cfg.get("output", {}))
    _check_keys("output", output, {"directory", "case", "results", "segy", "xdmf"})
    return Built(sim, output, base, names)


def load(path) -> Built:
    """Build a simulation from a YAML file."""
    import yaml

    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return build(cfg, path.parent)


def run(path, output_dir: Optional[str] = None, quiet: bool = False) -> Dict[str, Path]:
    """Build, save the case file, run and write the outputs of a YAML file.

    Returns the paths written (case, results, and any segy / xdmf files).
    """
    from . import post

    built = load(path)
    out = built.output
    directory = Path(output_dir) if output_dir else built.base_dir / out.get("directory", ".")
    directory.mkdir(parents=True, exist_ok=True)
    written = {"case": directory / out.get("case", "case.h5"), "results": directory / out.get("results", "results.h5")}
    sim = built.simulation
    sim.save(str(written["case"]))
    solver = sim.solver
    if not quiet:
        print(f"insem: {sim.dofs.n_elements} elements, dt = {solver.dt:.4g} s, {solver.n_steps} steps")
    sim.run()
    sim.write_results(str(written["results"]))
    for i, seg in enumerate(out.get("segy", [])):
        _check_keys("output.segy", seg, {"quantity", "component", "file", "receivers"}, ["quantity"])
        comp = seg.get("component", 0)
        name = seg.get("file", f"{seg['quantity']}_{comp}.sgy")
        written[f"segy{i}"] = post.write_segy(written["results"], directory / name, seg["quantity"], comp,
                                              seg.get("receivers"))
    if out.get("xdmf"):
        written["xdmf"] = post.write_xdmf(written["results"], directory / "snapshots.xdmf")
    if not quiet:
        for key, p in written.items():
            print(f"  {key}: {p}")
    return written
