# insem

Spectral element method (SEM) simulator for wave propagation in coupled
acoustic and elastic media. The primary target is sonic logging (monopole and
dipole, open and cased holes, layered formations).

The architecture, roadmap and open questions are in
[docs/DESIGN.md](docs/DESIGN.md).

## Status

Milestone **M0 (foundations)** is done:

| Area | What exists |
|---|---|
| Basis | GLL and GLJ(0,1) rules, Jacobi polynomials, Lagrange bases, derivative and interpolation matrices |
| Mesh | macro-mesh (Quad4/Quad9, Cartesian or axisymmetric) with topology, validation and boundary tags; forest of quadtrees with uniform refinement in Morton order |
| Geometry | leaf mappings, inverse Jacobians, geometry kinds (`rect`, `affine`, `curved`) |
| Operators | dof map with deterministic gather/assemble (no atomics, no colouring) |
| I/O | HDF5 wrapper; case files (`Simulation.save` / `Simulation.load`) |
| Interfaces | C++ library, `insem-run` executable, Python package (`insem`) with a rectangle mesh template |

There is no physics yet. The 2D acoustic/elastic solver is milestone M1.

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

# Axisymmetric (r, z) model with explicit radial breakpoints: borehole wall
# at r = 0.1 m, casing to 0.11 m, cement to 0.15 m.
m = mesh.rectangle([0.0, 0.1, 0.11, 0.15, 2.0], (0.0, 4.0), nz=20,
                   coords="axisymmetric", quadratic=True)
sim = insem.Simulation(m, insem.Discretization(order=4, uniform_level=1))
sim.setup()
print(sim.dofs.n_elements, sim.dofs.n_global, sim.geometry_kind_counts())
sim.save("case.h5")
```

```sh
insem-run case.h5
```

## Layout

```
cpp/include/insem/   public headers (basis, mesh, geometry, ops, io, sim)
cpp/src/             implementation
cpp/apps/            insem-run
python/insem/        Python package; python/src/bindings.cpp is the pybind11 module
tests/cpp/           Catch2 unit tests
tests/python/        pytest tests
docs/                design documents
```
