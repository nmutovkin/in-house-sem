# insem

Spectral element method (SEM) simulator for wave propagation in coupled
acoustic and elastic media. The primary target is sonic logging (monopole and
dipole, open and cased holes, layered formations).

The architecture, roadmap and open questions are in
[docs/DESIGN.md](docs/DESIGN.md).

## Status

Milestones **M0 (foundations)** and **M1 (2D Cartesian solver, CPU)** are done.

| Area | What exists |
|---|---|
| Physics | 2D isotropic elastic (P–SV) and acoustic (pressure formulation) in Cartesian coordinates, separate regions allowed (fluid–solid coupling is M2) |
| Discretisation | GLL spectral elements of order 1–8; macro-mesh (Quad4/Quad9) + uniform quadtree refinement; element blocks by geometry kind (`rect`, `affine`, `curved`); deterministic assembly |
| Time stepping | explicit central differences; stable Δt estimated by power iteration; discrete energy monitor (conserved to round-off) |
| Boundaries | free / rigid / Stacey absorbing per boundary tag (Stacey damping treated implicitly per point, so it does not limit Δt) |
| Sources | point force, moment tensor, acoustic monopole at arbitrary positions; Ricker, Gaussian, Gaussian derivative or sampled time functions |
| Receivers | displacement, velocity, acceleration (elastic); pressure and particle velocity (acoustic), at arbitrary positions |
| I/O | case files (full problem description), results files (traces, energy); `insem-run` |
| Python | `insem` package mirroring the C++ objects; `rectangle` and `layered` (curved interfaces) templates |

Verification (`tests/python/test_verification.py`): 2D acoustic and elastic
Green's functions (force, explosion, double couple) on straight and curved
meshes, fluid particle velocity, spectral convergence of an eigenmode, the
Rayleigh wave speed, Stacey absorption, and exact integrator identities.
Typical errors are 0.1–1 %; see the test file for the numbers.

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

`examples/layered_2d/run.py` is a fuller example: a two-layer elastic model
with a curved interface, a double-couple source and a vertical receiver line.

## Layout

```
cpp/include/insem/   public headers (basis, mesh, geometry, ops, physics, solver, io, sim)
cpp/src/             implementation (cpu/ holds the CPU element kernels)
cpp/apps/            insem-run
python/insem/        Python package; python/src/bindings.cpp is the pybind11 module
tests/cpp/           Catch2 unit tests
tests/python/        pytest tests
examples/            runnable examples
docs/                design documents
```
