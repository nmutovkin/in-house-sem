from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

import insem  # noqa: E402
from insem import cli, config, post  # noqa: E402

DATA = Path(__file__).parent / "data"

BASE = {
    "mesh": {"layered": {"x": [0, 4], "nx": 8, "interfaces": [-2, "0.1*sin(pi*x/4)", 1.5],
                         "nz": [4, 3], "regions": [1, 2], "quadratic": True}},
    "discretization": {"order": 3, "regions": [{"x": [1.5, 2.5], "z": [-0.5, 0.5], "level": 1}]},
    "materials": {1: {"medium": "elastic", "vp": 3.0, "vs": 1.7, "rho": 2.3},
                  2: {"medium": "acoustic", "vp": 1.5, "rho": 1.0}},
    "boundaries": {"bottom": "absorbing", "left": "absorbing", "right": "absorbing", "top": "free"},
    "sources": [{"type": "monopole", "position": [2.0, 0.8], "stf": {"type": "ricker", "f0": 2.0}}],
    "receivers": [
        {"line": {"start": [0.5, -0.5], "end": [3.5, -0.5], "n": 4}, "prefix": "g", "quantities": ["velocity"]},
        {"name": "h", "position": [1.0, 1.0], "quantities": ["pressure"]},
    ],
    "time": {"duration": 1.0, "record_every": 2, "snapshot_every": 25},
}


def test_build_matches_python_api():
    built = config.build(BASE)
    sim = built.simulation
    assert sim.discretization.regions == [(1.5, 2.5, -0.5, 0.5, 1)]
    assert sim.boundaries == {1: "absorbing", 2: "absorbing", 3: "free", 4: "absorbing"}
    assert [r.name for r in sim.receivers] == ["g0", "g1", "g2", "g3", "h"]
    assert sim.receivers[3].position[0] == pytest.approx(3.5)
    assert sim.sources[0].kind == "monopole"
    # The curved interface formula is honoured exactly at the nodes.
    nodes = sim.mesh.nodes
    on_interface = np.isclose(nodes[:, 1], 0.1 * np.sin(np.pi * nodes[:, 0] / 4))
    assert on_interface.sum() >= 17


def test_cli_run_writes_all_outputs(tmp_path):
    cfg = dict(BASE, output={"segy": [{"quantity": "velocity", "component": "z", "file": "vz.sgy"}], "xdmf": True})
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert cli.main(["run", str(path), "-o", str(tmp_path / "out")]) == 0
    out = tmp_path / "out"
    for name in ("case.h5", "results.h5", "vz.sgy", "snapshots.xdmf"):
        assert (out / name).exists()
    res = post.read_results(out / "results.h5")
    seg = post.read_segy(out / "vz.sgy")
    np.testing.assert_array_equal(seg.data[2], res.trace("g2", "velocity")[:, 1].astype(np.float32))
    # The case file reproduces the run exactly.
    sim = insem.Simulation.load(str(out / "case.h5"))
    sim.run()
    np.testing.assert_array_equal(sim.solver.trace("h", "pressure"), res.trace("h", "pressure"))
    assert cli.main(["info", str(out / "case.h5")]) == 0


def test_cli_build(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(BASE))
    assert cli.main(["build", str(path), "-o", str(tmp_path / "c.h5")]) == 0
    assert insem.Simulation.load(str(tmp_path / "c.h5")).time.duration == 1.0


def test_gmsh_names_and_sampled_stf(tmp_path):
    np.savetxt(tmp_path / "wavelet.txt", np.sin(np.linspace(0, np.pi, 50)) ** 2)
    (tmp_path / "mesh.msh").write_bytes((DATA / "layered_q4_v41.msh").read_bytes())
    cfg = {
        "mesh": {"gmsh": {"file": "mesh.msh"}},
        "materials": {"rock": {"medium": "elastic", "vp": 3.0, "vs": 1.7, "rho": 2.3},
                      "water": {"medium": "acoustic", "vp": 1.5, "rho": 1.0}},
        "boundaries": {"bottom": "absorbing", "sides": "absorbing", "top": "free"},
        "sources": [{"type": "monopole", "position": [2.0, 2.8],
                     "stf": {"type": "samples", "file": "wavelet.txt", "dt": 0.01}}],
        "time": {"n_steps": 10},
    }
    path = tmp_path / "g.yaml"
    path.write_text(yaml.safe_dump(cfg))
    sim = config.load(path).simulation
    assert set(sim.materials) == {1, 2}
    assert sim.boundaries == {11: "absorbing", 12: "free", 13: "absorbing"}
    assert sim.sources[0].stf.kind == "samples"


@pytest.mark.parametrize("bad,match", [
    ({"mesh": {"rectangle": {"x": [0, 1], "z": [0, 1], "nx": 1, "nz": 1, "colour": 3}}}, "unknown key"),
    ({"time": {"duration": 1.0, "steps": 3}}, "unknown key"),
    ({"materials": {7: {"medium": "elastic", "vp": 3.0, "vs": 1.0, "rho": 1.0}}}, None),
    ({"boundaries": {"floor": "absorbing"}}, "unknown boundary"),
    ({"sources": [{"type": "laser", "position": [0, 0], "stf": {"type": "ricker", "f0": 1}}]}, "unknown type"),
])
def test_invalid_configs(bad, match):
    cfg = {"mesh": {"rectangle": {"x": [0, 1], "z": [0, 1], "nx": 2, "nz": 2}},
           "materials": {0: {"medium": "elastic", "vp": 3.0, "vs": 1.7, "rho": 2.3}},
           "time": {"n_steps": 5}}
    cfg.update(bad)
    if match is None:  # a material for a region that does not exist: caught when the solver is built
        with pytest.raises(insem.InsemError):
            config.build(cfg).simulation.run()
        return
    with pytest.raises(config.ConfigError, match=match):
        config.build(cfg)


@pytest.mark.parametrize("formula", ["__import__('os').system('true')", "x.__class__", "open('f')", "y + 1",
                                     "(lambda: 1)()"])
def test_formulas_cannot_run_code(formula):
    with pytest.raises(config.ConfigError, match="not allowed"):
        config._formula(formula)


def test_formulas_evaluate():
    f = config._formula("-150 + 30*sin(2*pi*x/800) + sqrt(abs(x))")
    x = np.array([0.0, 200.0])
    np.testing.assert_allclose(f(x), -150 + 30 * np.sin(2 * np.pi * x / 800) + np.sqrt(x))


def test_borehole_config():
    cfg = yaml.safe_load((Path(__file__).parents[2] / "examples" / "yaml" / "cased_hole.yaml").read_text())
    built = config.build(cfg)
    sim = built.simulation
    assert sim.mesh.coords == "axisymmetric"
    assert built.names["regions"] == {"mud": 1, "casing": 2, "cement": 3, "shale": 4, "sand": 5}
    assert sim.boundaries == {1: "absorbing", 2: "absorbing", 3: "absorbing"}
    assert sim.materials[2].vs == 3200.0
    assert len(sim.receivers) == 8 and sim.receivers[0].position[0] == 0.0
    with pytest.raises(config.ConfigError, match="axisymmetric"):
        config.build(dict(cfg, mesh=dict(cfg["mesh"], coords="cartesian")))
