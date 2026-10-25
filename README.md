# insem

Spectral element method (SEM) simulator for wave propagation in coupled
acoustic and elastic media. The primary target is sonic logging (monopole and
dipole, open and cased holes, layered formations).

The architecture, roadmap and open questions are in
[docs/DESIGN.md](docs/DESIGN.md).

## Status

Milestones **M0–M3** are done: a verified 2D acoustic–elastic solver on CPU
with fluid–solid coupling and adaptive meshes, in Cartesian (x, z) and
**axisymmetric (r, z)** form, with a borehole template, a discrete-wavenumber
reference and STC for **monopole sonic logging** in open and cased holes.

| Area | What exists |
|---|---|
| Physics | isotropic elastic and acoustic (pressure formulation), fluid–solid coupling; Cartesian 2D (P–SV) and **axisymmetric** (true 3D fields for axisymmetric models: monopole sonic logging) |
| Meshes | templates (`rectangle`, `layered` with curved interfaces, **`borehole`** with annuli, beds and radial grading) or Gmsh (`.msh` 2.2/4.1); adaptive quadtree refinement (points per wavelength at f_max, boxes) with 2:1 balance and hanging nodes |
| Discretisation | GLL spectral elements of order 1–8; GLJ points along r in elements on the axis; element blocks by geometry kind; deterministic assembly |
| Time stepping | explicit central differences; stable Δt from power iteration (coupled operator included); discrete energy monitor |
| Boundaries | free / rigid / Stacey absorbing per boundary tag; symmetry axis |
| Sources | point force, moment tensor, acoustic monopole (3D point sources on the axis, ring sources off it); Ricker, Gaussian, Gaussian derivative or sampled time functions |
| Receivers | displacement, velocity, acceleration (elastic); pressure, particle velocity (acoustic) |
| Output | case and results files (HDF5), snapshots, ParaView (XDMF) and SEG-Y rev 2 export, **slowness–time coherence** |
| References | `insem.reference.borehole_axis_pressure`: discrete-wavenumber solution for a monopole in a fluid-filled borehole with any fluid/solid annuli |
| Interfaces | Python API, YAML configs and the `insem` command, C++ `insem-run` |

Verification (all against independent references; see `tests/python`):

| Check | Error |
|---|---|
| 2D Green's functions (force, explosion, double couple), straight and curved meshes | 0.1–1.3 % |
| Lamb's problem (wavenumber integration) | 1.6 % (order 4), 0.3 % (order 6) |
| Fluid–solid R/T at normal incidence; Scholte speed; Rayleigh speed | 4 digits; 0.16 %; 0.2 % |
| **Axisymmetric vs 3D analytic**: fluid point source; elastic vertical force, explosion | 0.05–0.7 %; 0.1–0.6 % |
| **Borehole monopole vs discrete-wavenumber reference**: open hole; cased hole (steel + cement) | 0.2–0.9 %; 0.2–0.6 % |
| STC picks of SEM vs reference arrays (P, S, Stoneley) | identical |
| Energy conservation (also refined, axisymmetric); refinement-interface reflection | round-off; −66 dB at ~7.5 points/wavelength |

## Building

Requirements: CMake ≥ 3.24, a C++17 compiler, HDF5 (C library and headers),
Catch2 v3 (for the tests), and optionally OpenMP. On Ubuntu:

```sh
sudo apt-get install libhdf5-dev catch2
```

C++ library, tests and `insem-run`:

```sh
cmake -S . -B build -DINSEM_REAL=float      # or double
cmake --build build -j
ctest --test-dir build
```

Main CMake options:

| Option | Default | Meaning |
|---|---|---|
| `INSEM_REAL` | `float` | type of solver fields (`float` or `double`) |
| `INSEM_ENABLE_OPENMP` | `ON` | OpenMP for the CPU backend |
| `INSEM_STRICT_FP` | `OFF` | disable FMA contraction, for bitwise CPU/CUDA comparison |
| `INSEM_ENABLE_CUDA`, `INSEM_ENABLE_MPI` | `OFF` | reserved (M5, M8) |

Python package (builds the same C++ core through scikit-build-core):

```sh
pip install ".[test]"        # or: pip install --no-build-isolation -e .
pytest
```

## Example

```python
import insem
from insem import mesh

m = mesh.rectangle((-4.0, 4.0), (-4.0, 4.0), nx=32, nz=32)
sim = insem.Simulation(m, insem.Discretization(order=4))
sim.set_material(0, insem.Material.acoustic(vp=1500.0, rho=1000.0))
for tag in (mesh.BOTTOM, mesh.RIGHT, mesh.TOP, mesh.LEFT):
    sim.set_boundary(tag, "absorbing")
sim.add_source(insem.Source.monopole(0.0, 0.0, insem.ricker(400.0)))
sim.add_receiver("h1", 1.5, 0.0, ["pressure"])
sim.set_time(duration=4e-3)
sim.run()
pressure = sim.solver.trace("h1", "pressure")   # (n_samples, 1)
sim.save("case.h5")                             # re-run with: insem-run case.h5
```

The same through a YAML file and the command line:

```sh
insem run examples/yaml/water_over_rock.yaml   # case, results, SEG-Y, XDMF
insem info water_over_rock_out/case.h5
insem-run water_over_rock_out/case.h5          # C++ executable, no Python needed
```

More examples:

| Example | What it shows |
|---|---|
| `examples/borehole/run.py` | monopole sonic log in open and cased holes: mesh, snapshots, array waveforms vs the discrete-wavenumber reference, STC |
| `examples/yaml/cased_hole.yaml` | cased hole crossing a sand bed, as a YAML configuration |
| `examples/layered_2d/run.py` | two-layer elastic model with a curved interface, double-couple source, receiver line |
| `examples/gmsh_seabed/run.py` | Gmsh mesh (curved seabed, buried lens), wavelength-based refinement, fluid–solid coupling, snapshot and gather plots; `make_mesh.py` regenerates the mesh with Gmsh |
| `examples/verification/plot_verification.py` | SEM against Lamb's problem and the analytic double-couple solution |
| `examples/yaml/water_over_rock.yaml` | the same kind of model as a YAML configuration |

## Layout

```
cpp/include/insem/   public headers (basis, mesh, geometry, ops, physics, solver, io, sim)
cpp/src/             implementation (cpu/ holds the CPU element kernels)
cpp/apps/            insem-run
python/insem/        Python package (mesh templates, gmsh reader, config/CLI, post-processing);
                     python/src/bindings.cpp is the pybind11 module
tests/cpp/           Catch2 unit tests
tests/python/        pytest tests, analytic references, Gmsh fixtures
examples/            runnable examples
docs/                design documents
```
