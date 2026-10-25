# In-house SEM: Architecture & Roadmap

Status: **draft v2 for review**. Nothing here is implemented yet.

A spectral element method (SEM) simulator for wave propagation in coupled
acoustic (fluid) and elastic (solid) media.

The primary target is **sonic logging**: monopole and dipole sources in open
and cased holes (casing + cement) crossing layered formations. Passive
monitoring (microseismic, DAS) is a secondary target and is kept in the design.

---

## 0. Originality and clean-room policy

This is an independent design, **not a port or restructuring of SPECFEM**
(or any other SEM code). Its main departures from the SPECFEM family are:

| Topic | Typical SPECFEM approach | This code |
|---|---|---|
| Fluid unknown | displacement potential χ | **pressure p** (§2.2) |
| Operator structure | one global loop over all elements; colouring (CPU) or atomics (GPU) to add up element contributions | **element blocks** + gather → local kernel → **deterministic scatter**: no colouring, no atomics, reproducible results (§7.3) |
| Mesh refinement | conforming "doubling" bricks built by hand | **forest of quadtrees** (octrees in 3D) with λ-driven adaptive refinement and hanging-node constraints (§6) |
| Workflow | separate mesher → database → solver programs | **library-first**: one `Simulation` object; the case file is its serialisation; partitioning happens at load (§7.1) |
| Time stepping | global Newmark | pluggable integrators; **multi-rate local time stepping** planned (§3.2) |
| Partitioning | external graph partitioner | **space-filling curve** over the forest, weighted by cost (§7.4) |

**Clean-room rule.** SPECFEM is GPL-3. p4est, which provides the
forest-of-trees ideas, is GPL-2. No code from either, nor from any other
copyleft code, is copied, translated or used as a template. Implementation
works only from published papers and textbooks. Running other codes as
**black-box references** to compare outputs is allowed. All dependencies must be
permissively licensed (BSD, MIT or Apache) or LGPL linked dynamically.

The core SEM mathematics is standard textbook material and is used as such:
GLL collocation, diagonal mass matrix, Stacey and PML absorbing boundaries.
References include Patera 1984, Maday & Rønquist 1990, Komatitsch & Vilotte
1998, Fichtner 2011, and Deville, Fischer & Mund 2002.

---

## 1. Decisions so far

| Topic | Decision |
|---|---|
| Name | `insem` (C++ namespace, Python package, `insem-run` executable) |
| Primary application | sonic logging: monopole + dipole; open and cased hole (casing, cement); layered formations / bed boundaries |
| Language | C++17 core, Python bindings (pybind11) |
| Dimensions | 2D first (Cartesian **and** axisymmetric r–z); designed so 3D (hex, octree) slots in |
| Physics v1 | isotropic acoustic (**pressure**) + isotropic elastic + **fluid–solid coupling** |
| Physics later | anisotropy (VTI/TTI), attenuation, topography, azimuthal Fourier (2.5D) |
| Meshing | conforming **macro-mesh** (in-house layered/borehole templates or Gmsh import) refined by a **forest of quadtrees** with hanging nodes |
| Operators | matrix-free, element blocks, deterministic gather/scatter |
| Time stepping | pluggable integrators (central difference first), local time stepping planned |
| Absorbing BC | Stacey first, C-PML next |
| Parallelism | native CUDA + separate CPU (OpenMP) code, MPI with SFC partitioning |
| Precision | templated on real type; `float` default, `double` for verification |
| User interface | Python API and C++ API over one `Simulation` object; YAML is a thin layer over the Python API |
| Output | HDF5 + SEG-Y export + XDMF for ParaView |
| Units | SI (m, s, kg/m³, Pa) |

---

## 2. Physics and weak forms

### 2.1 Elastic (solid)

$$\rho\,\ddot{\mathbf u} = \nabla\cdot\boldsymbol\sigma + \mathbf f,\qquad
\boldsymbol\sigma = \lambda\,(\nabla\cdot\mathbf u)\,\mathbf I + 2\mu\,\boldsymbol\varepsilon,\qquad
\boldsymbol\varepsilon = \tfrac12(\nabla\mathbf u + \nabla\mathbf u^T)$$

Weak form: $\int \rho\,\mathbf w\cdot\ddot{\mathbf u} = -\int \nabla\mathbf w:\boldsymbol\sigma + \int \mathbf w\cdot\mathbf f + \oint \mathbf w\cdot\mathbf t$.
The free surface is the natural condition (**t** = 0).

The constitutive law sits behind one point-wise function. Anisotropy and
attenuation later replace that function and add state; they do not change the
operator structure.

### 2.2 Acoustic (fluid): pressure formulation

We solve directly for pressure, which is the quantity that hydrophones and
borehole sonic receivers measure. From ρ ü = −∇p and p = −κ ∇·u:

$$\tfrac{1}{\kappa}\,\ddot p = \nabla\cdot\!\left(\tfrac{1}{\rho}\nabla p\right) + \ddot Q\,\delta_{x_s}$$

Here Q(t) is the injected volume (monopole source).

Weak form, with **n**_f the outward normal of the fluid:
$$\int \tfrac1\kappa\, w\,\ddot p = -\int \tfrac1\rho\nabla w\cdot\nabla p \;-\; \oint w\,(\mathbf n_f\cdot\ddot{\mathbf u}) \;+\; w(x_s)\,\ddot Q$$
This follows because (1/ρ) **n**·∇p = −**n**·ü.

* Pressure-release surface: Dirichlet p = 0.
* Rigid wall: natural condition (**n**·ü = 0).
* Variable density inside the fluid (mud, brine, gas layers) is handled exactly by the 1/ρ inside the divergence.
* Fluid particle velocity (for vector hydrophones and DAS in fluid) comes from
  integrating v̇ = −∇p/ρ in time at the receiver point. This is cheap and local.

### 2.3 Fluid–solid coupling

The interface conditions are continuity of normal acceleration and traction
**σ** **n**_s = −p **n**_s, where **n**_s = −**n**_f.

* **Solid side:** the traction term becomes $\oint \mathbf w\cdot(p\,\mathbf n_f)$. It needs p.
* **Fluid side:** the boundary term above, $-\oint w\,\mathbf n_f\cdot\ddot{\mathbf u}_s$. It needs ü of the solid.

Explicit ordering within one acceleration evaluation (no iteration):
1. **Solid:** internal forces + coupling traction from the current pressure pⁿ → üⁿ.
2. **Fluid:** internal forces + coupling term from the just-computed üⁿ → p̈ⁿ.

The ordering lives inside the coupled system's `evaluate_acceleration`, so every
time integrator (§3.2) inherits it.

Interfaces are stored as their own objects: (fluid face, solid face, quadrature
points, normal, weight × surface Jacobian). Both sides of a fluid–solid
interface must be refined to the same level, so the interface points match one
to one. The mesh balancing step (§6.2) enforces this.

### 2.4 Axisymmetric (r–z) formulation, for boreholes

A borehole with a monopole source on its axis is a 3D problem with rotational
symmetry. Solving it on an (r, z) mesh gives the true 3D wavefield (P and S
head waves, Stoneley, pseudo-Rayleigh) at 2D cost.

* Volume element: dΩ = 2π r dr dz. Every integral carries an r weight.
* Elastic unknowns (u_r, u_z), with hoop strain ε_θθ = u_r / r. Torsional u_θ decouples and is optional.
* Fluid unknown: p(r, z).
* **Axis (r = 0):** elements touching the axis use **Gauss–Lobatto–Jacobi
  (0,1)** points in r. This absorbs the r weight and removes the 1/r
  singularity (Bernardi, Dauge & Maday 1999; Nissen-Meyer et al. 2007). The axis
  condition is u_r = 0, with u_r/r → ∂u_r/∂r on the axis.
* Axisymmetry is a **geometry kind**, not a separate solver (§7.3). Axial and
  off-axis axisymmetric elements are separate element-block types with their own
  quadrature and hoop terms.

**Limitation:** pure axisymmetry (m = 0) only allows on-axis sources with
axisymmetric radiation. That covers the monopole sonic source, an explosion
and a vertical force. The dipole needs §2.5. General off-axis microseismic
moment tensors need full 3D, so v1 microseismic work is done in 2D Cartesian (P–SV).

### 2.5 Azimuthal Fourier modes (2.5D): dipole logging

The geometry stays axisymmetric (the borehole, casing, cement and horizontal
beds), but the source does not have to be. Fields are expanded in azimuthal
order m, and each mode is an independent 2D problem on the **same r–z mesh**:

* Elastic, for a source polarised along x:
  u = (u_r^m(r,z) cos mθ, u_θ^m(r,z) sin mθ, u_z^m(r,z) cos mθ). That is three
  unknowns per point, and the strain picks up m/r terms from ∂/∂θ.
* Acoustic: p = p^m(r,z) cos mθ, which adds a −m²/r² term (weighted by r²/ρ).
* **m = 0** is the monopole (§2.4). **m = 1** is the dipole: flexural waves and
  shear logging in slow formations. **m = 2** (quadrupole) uses the same code
  path and can be added later.
* Axis conditions depend on the mode, and the GLJ axial quadrature handles the
  1/r and 1/r² terms (Nissen-Meyer et al. 2007). For m = 1: u_z = 0,
  u_r^1 + u_θ^1 = 0 and p^1 = 0 on the axis (the exact sign depends on the
  cos/sin convention and is fixed in the implementation notes).
* Fluid–solid coupling, Stacey boundaries and the element-block structure carry
  over per mode. Each m is a separate geometry/physics block family.
* A dipole source in the fluid is a point force density **f**:
  (1/κ) p̈ = ∇·(1/ρ ∇p) − ∇·(**f**/ρ), so it enters through ∇w. Receivers
  output the mode amplitude, and the azimuthal pattern (cos θ) is applied in post-processing.

### 2.6 Later physics (designed for, not built)

* Attenuation: standard linear solids with memory variables per GLL point.
* Anisotropy: a full C_ij per GLL point. VTI in a vertical well remains
  axisymmetric and fits §2.4–2.5; deviated wells and TTI need 3D.
* Topography and curved interfaces: already supported through curved macro-elements (§6).
* A logging-tool body: just another elastic annulus in the borehole template.

---

## 3. Discretisation and time integration

### 3.1 Space

* **Basis:** Lagrange polynomials on GLL points, degree N (default **N = 4**).
  GLJ(0,1) is used in r on axial elements.
* **Quadrature:** collocated, which gives a diagonal element mass matrix. Hanging
  nodes need care (§6.3).
* **Kernels:** tensor-product sum factorisation with a 1D derivative matrix
  **D** of size (N+1)×(N+1). N is a compile-time template parameter (instantiated for N = 2…8).
* **Geometry kinds,** each with its own block type:
  * `rect`: axis-aligned rectangles. The metric is two scalars per element, and
    most quadtree leaves in layered and borehole models are this kind, so it is
    the fast path.
  * `affine`: parallelograms. One constant 2×2 metric per element.
  * `curved`: a per-point metric precomputed from the parent macro-element's
    mapping (bilinear or quadratic).
  * `axisym-*`: the variants above with an r weight, and GLJ on the axis.

  Storing the metric only where it varies cuts memory traffic, which is the
  real bottleneck of SEM kernels on both CPU and GPU.
* **Materials** are sampled at GLL points at load time from a material model
  (layers, gradients, gridded volumes). The case file stores the model, not
  per-point arrays.
* **Resolution rule:** about 5 GLL points per minimum wavelength, so h ≲ N·λ_min/5.
  This is the refinement criterion the mesher uses (§6).

### 3.2 Time

* `TimeIntegrator` interface acting on a semi-discrete coupled system
  `M ẍ = R(x, ẋ, t)`. The system object owns the physics ordering (§2.3) and the
  constraints (§6.3).
* **v1:** explicit central difference (leapfrog), with Δt from a per-element CFL estimate.
* **Later:** low-storage Runge–Kutta (LDDRK) for C-PML auxiliary equations.
* **Local time stepping (planned).** Each element gets a level ℓ with
  Δt_ℓ = Δt₀ / 2^ℓ from its own CFL limit. The integrator is the leapfrog-based
  multi-rate scheme of Diaz & Grote (2009), which is explicit and conserves a
  discrete energy. This fixes the borehole bottleneck: tiny elements at the
  borehole wall no longer dictate Δt for the whole formation. The level
  structure lines up with quadtree depth, and element blocks are also split by
  LTS level so each level's update is one batched kernel launch.

---

## 4. Boundaries

| Boundary | Elastic | Acoustic (pressure) |
|---|---|---|
| Free / pressure-release | natural (t = 0) | Dirichlet p = 0 |
| Rigid | Dirichlet u = 0 | natural (n·∇p = 0) |
| Stacey (v1) | t = −ρ[c_p (v·n)n + c_s (v − (v·n)n)] | n·∇p = −ṗ / c, giving boundary term −∮ w ṗ/(ρc) |
| C-PML (M6) | auxiliary-field layer, element blocks of PML kind | same |
| Axis (axisym) | u_r = 0 + GLJ quadrature | GLJ quadrature |

Boundaries are face sets tagged on the macro-mesh. Refined leaves inherit the tags.

---

## 5. Sources and observers

Everything that injects into the system is a **Forcing**. Everything that reads
from the system is an **Observer**. Both are located at arbitrary positions:
an inverse map gives the reference coordinates, and Lagrange interpolation (or
its derivative) gives the weights.

**Forcings**
* Point force and **moment tensor** (solid) for microseismic. 2D Cartesian
  uses M_xx, M_zz and M_xz; axisymmetric mode allows on-axis axisymmetric parts only.
* Volume-injection monopole (fluid) for borehole sonic sources (§2.2).
* Dipole: a point force in the fluid or solid, run in mode m = 1 (§2.5).
* Source time functions: Ricker, Gaussian and derivatives, smoothed step, or a user array.
* Any number of sources, for multi-event microseismic catalogues.

**Observers** (a common interface: `prepare`, `record(step)`, `flush`)
* Point receivers. Solid: u, v, a. Fluid: p, and particle velocity
  (§2.2). A receiver placed exactly on an interface must say which side it records.
* **DAS fibers.** A polyline (a 3D curve later) with channel spacing and gauge
  length L. Output is axial strain or strain rate, either point-wise
  **t**·ε·**t** or gauge-averaged [**u**(s+L/2) − **u**(s−L/2)]·**t** / L (the
  default). The averaging is done on the fly.
* Snapshots (decimated) and an **energy monitor** (kinetic + strain + acoustic
  energy). The energy monitor is a standard stability and diagnostic tool.

Observers record into device-side buffers that are flushed to the host every K steps.

---

## 6. Meshing: macro-mesh + forest of quadtrees

### 6.1 Macro-mesh (conforming, coarse)
The macro-mesh honours geometry. Every material interface, and in particular
every fluid–solid interface, lies on macro-element faces. Sources of macro-meshes:
* **In-house templates (Python):** a layered model (layers bounded by z_k(x)
  or z_k(r)), and a **borehole template**: a fluid column r < a, then optional
  steel casing and cement annuli (bonded = shared solid–solid faces; a
  debonded fluid micro-annulus is just a thin fluid annulus), then the
  formation, split into horizontal beds at given depths.
* Thin annuli (casing about 1 cm thick, with v_p ≈ 5.9 km/s) produce tiny
  elements. They are the main reason local time stepping (§3.2) is on the roadmap.
* **Gmsh import:** `.msh` v4, quad4 and quad9 (hex later), with physical groups
  mapped to materials and boundary tags.

Macro-elements may be curved. Their mapping is inherited by every child.

### 6.2 Adaptive refinement (C++ core, exposed to Python)
* Each macro-element is the root of a quadtree. Leaves are the computational elements.
* **Refinement criterion:** refine while the leaf violates h ≤ N·λ_min/ppw
  (λ_min sampled from the material model at the leaf's GLL points). Optional
  user refinement regions can be added (around sources, receivers, the borehole wall).
* **Balance:** 2:1 across all faces, including faces between macro-elements, so
  an edge has at most one hanging level. Fluid–solid faces are refined
  identically on both sides (§2.3).
* The result is one generic description: leaf list + per-leaf geometry kind +
  hanging-face table. Everything downstream (operators, partitioning, LTS
  levels) consumes it.
* This replaces hand-built doubling layers. Large velocity contrasts (water vs
  rock, borehole fluid vs fast formation) are meshed automatically.
* 3D later: an octree over hex macro-elements, with the same data model.

### 6.3 Hanging nodes in the operators
* GLL points on the fine side of a coarse–fine face are **constrained**: they
  are interpolated from the coarse face's GLL points (**u**_h = C **u**_master).
* The constraints are folded into the **gather/scatter operators** (§7.3).
  Gather applies C, and scatter applies Cᵀ. The local kernels never know about hanging nodes.
* **Mass:** the assembled Cᵀ M C is not diagonal on master points of hanging
  faces. It is row-sum lumped to keep explicit updates. That is a numerical
  choice and needs verification: measure spurious reflection at refinement
  interfaces, the convergence rate, and energy behaviour (§9).
* **Fallback** if the reflection is unacceptable: conforming 2:1 transition
  templates for the affected faces, which need no hanging nodes. The data model supports both.

### 6.4 Quality and diagnostics
Checks for Jacobian sign and condition, points per wavelength per leaf, the CFL
Δt per leaf (and the resulting LTS level histogram), and an estimated
cost per partition.

---

## 7. Software architecture

### 7.1 Library-first run model

```
          Python API / YAML                     C++ API
                 │                                 │
                 └──────────────┬──────────────────┘
                                ▼
                          Simulation
   ┌─────────┬──────────┬───────┴────┬───────────┬──────────┬──────────┐
 MacroMesh  Material   Refinement   Forcings   Observers   Integrator  Backend
 + tags     model      policy                                          (cpu|cuda)
                                │
                  sim.save("case.h5") / Simulation::load
                                ▼
            insem-run case.h5   (tiny executable: load → partition → run)
```

* There is **one object model** in C++, mirrored in Python. Interactive runs call
  `sim.run()` in-process and get NumPy views of the results.
* The **case file is a serialisation** of the `Simulation`: macro-mesh,
  material model, refinement policy, forcings, observers and run parameters. It
  is independent of the rank count, backend and N. Refinement, material
  sampling and partitioning all happen at load time, on the actual parallel layout.
* `insem-run` contains no logic of its own: it calls `Simulation::load(...).run()`
  under `mpirun`. Compute nodes need no Python.

### 7.2 Repository layout (proposed)

```
cpp/include/insem/
  basis/        gll, glj, lagrange, derivative matrices
  geometry/     mappings, geometry kinds, inverse map
  mesh/         macro_mesh, forest (quadtree), balance, constraints, tags
  model/        material models, sampling
  ops/          dof_map (gather/scatter incl. constraints), element blocks,
                operator interface
  physics/      pointwise.hpp  ← shared HD-inline physics (CPU + CUDA)
                elastic, acoustic, coupling, stacey
  forcing/      point force, moment tensor, monopole, STFs
  observe/      receivers, das, snapshots, energy
  time/         integrator interface, leapfrog, (lts, lddrk later)
  parallel/     sfc partition, halo exchange
  io/           case file, results file (HDF5)
  sim/          Simulation
cpp/src/{cpu,cuda,...}
cpp/apps/insem_run.cpp
python/insem/   api mirror, templates (layered, borehole), gmsh_io, yaml,
                postproc (segy, xdmf), plotting
tests/{cpp,python}   examples/   docs/
```

`insem` is a placeholder name.

### 7.3 Operators: element blocks + deterministic gather/scatter

Applying the stiffness/coupled operator:
```
for each element block B (homogeneous: physics × geometry kind × N × LTS level):
    x_e  = gather_B(x)            # element-local copies, applies hanging constraints C
    f_e  = kernel_B(x_e)          # sum-factorised local operator, no branching
f = scatter(f_e over all blocks)  # Cᵀ, transpose of gather
```
* **Blocks are homogeneous,** so each kernel is a template instantiation with
  no runtime branching. This is what makes adding axisymmetric, PML,
  attenuation or 3D blocks additive rather than invasive.
* **Deterministic scatter.** The transpose map (for each global point, the
  ordered list of contributing element-local slots and constraint weights) is
  precomputed as a CSR table. The scatter is one pass over global points,
  summing in that fixed order: one CPU thread or one GPU thread per point, and
  **no atomics, no colouring**. The cost is storing the element-local force
  array, about 1.5× the global array for N = 4 in 2D. That is acceptable.
* **Reproducibility:**
  * Each backend is always run-to-run deterministic, independent of thread or block counts.
  * In a `strict` build (FMA contraction disabled on both compilers, identical
    loop order in kernels), **CPU and CUDA are bitwise identical**. CPU-vs-CUDA
    tests then compare exactly, which is the strongest guard against the two
    code paths drifting apart.
  * In normal builds the comparison uses a tolerance.
* **Shared point-wise physics.** Everything at a single point (stress, traction,
  coupling, Stacey, interpolation weights, DAS projection) is in
  `physics/pointwise.hpp` as `INSEM_HD inline` functions. Both backends call
  these. The two code paths differ only in loop and thread mapping.
* **CUDA mapping:** several elements per thread block, one thread per point, and
  **D** in shared memory. `rect` and `affine` blocks read almost no geometry
  data, which helps bandwidth.

### 7.4 MPI and partitioning
* **Space-filling-curve partitioning** of the forest leaves (Morton or
  Hilbert order across macro-elements), weighted by per-leaf cost (physics,
  N, and LTS level × substeps). This needs no graph-partitioner dependency and
  gives contiguous, cache-friendly local orderings. Repartitioning is cheap.
* Shared global points are summed across ranks after the local scatter. The
  inner/outer split overlaps communication with computation.
* For GPUs, halo packing happens on the device, with CUDA-aware MPI when
  available and staging through pinned host buffers otherwise.
* Observers belong to the rank that owns their element. Fluid–solid interfaces
  may cross ranks.

### 7.5 Build and dependencies
* CMake with `INSEM_ENABLE_CUDA`, `INSEM_ENABLE_MPI`, `INSEM_ENABLE_OPENMP`,
  `INSEM_REAL`, `INSEM_STRICT_FP`, and the instantiated N list.
* Required: HDF5. Optional: CUDA ≥ 12, MPI. No partitioner and no mesh library is required.
* Python: pybind11, numpy, h5py, pyyaml, optional gmsh. Built with scikit-build-core.
* Tests: Catch2 and pytest. CI here is CPU only; CUDA and MPI tests run on your hardware.

---

## 8. Output

HDF5 `results.h5`:
```
/meta                    run parameters, dt, nt, units, version, git hash, case hash
/receivers/<name>/<quantity>          [nt, ncomp]   (+ position, medium, components)
/das/<fiber>/{strain|strain_rate}     [nt, nchannels] (+ gauge length, positions, tangents)
/energy                               [nt_energy, {kinetic, elastic, acoustic}]
/snapshots/<field>/<step>             (optional, decimated)
```
* SEG-Y rev 2 export (receiver gathers, DAS channels).
* Sonic post-processing (Python): slowness–time coherence (STC), dispersion
  extraction (matrix pencil / Prony) across a receiver array, and overlays of
  modal dispersion curves computed by the reference solver (§9).
* XDMF sidecar files over the snapshots for ParaView, written as leaf-level
  high-order cells subdivided for display.

---

## 9. Verification plan

| Test | Reference |
|---|---|
| GLL/GLJ points, weights, derivative matrices | closed forms / high-precision tables |
| Gather/scatter: Cᵀ is the exact transpose of C; constraints reproduce polynomials | algebraic identity tests |
| 2D homogeneous acoustic (pressure), point source | analytic 2D Green's function ⊛ STF |
| 2D homogeneous elastic, force / moment tensor | analytic 2D elastic Green's function |
| Lamb's problem | Garvin / Lamb analytic |
| Fluid–solid interface (reflection, Scholte wave) | analytic plane-wave coefficients; Gar6more2D as black box |
| Spectral convergence (N, h), conforming **and** refined meshes | expected rates |
| Spurious reflection at refinement interfaces (hanging nodes) | must stay below a set threshold (e.g. −60 dB at 5 ppw) |
| Energy conservation, closed box, with and without LTS | constant to round-off (leapfrog / Diaz–Grote energy) |
| Stacey reflection vs incidence angle | larger-domain reference |
| Axisym point sources on axis | analytic 3D spherical waves |
| Borehole monopole and dipole, open and cased hole (homogeneous formation) | in-house discrete-wavenumber reference with radial propagator matrices (Tsang & Rader 1979; Kurkjian & Chang 1986; Tubman et al. 1984 for cased holes) |
| Stoneley and flexural dispersion | modal root-finding of the borehole dispersion relation vs SEM-extracted dispersion |
| DAS gauge-length response | strain from analytic displacement |
| CPU vs CUDA (strict build), 1 vs N ranks | **bitwise** identical |

---

## 10. Open questions for review

Resolved: the name is `insem`. The target is sonic logging (monopole + dipole,
cased hole, layered formations). No anisotropy or attenuation early.

Still open:
1. **Typical runs:** frequency band (monopole about 1–20 kHz, dipole about
   0.5–5 kHz?), borehole and casing dimensions, source–receiver offsets and array
   length, and record length. These set the domain size and Δt, and decide when
   C-PML and LTS become urgent.
2. **Tool:** do we ever need the tool body (a rigid or elastic mandrel), or
   is a free fluid column enough? It is cheap to add as an annulus (§2.6).
3. **Run volume:** many small 2.5D runs (parameter sweeps, inversion) or a
   few big ones? Many small runs favour batching several independent
   simulations on one GPU, rather than MPI.
4. **Hardware:** GPU models or compute capability, MPI, CUDA-aware MPI, and whether a GPU CI runner is available.
5. **Dependencies:** confirm permissive-only (BSD, MIT, Apache; LGPL linked dynamically).
6. **Gmsh:** is the Gmsh Python package acceptable at runtime, or should we only read `.msh` files?

---

## 11. Roadmap

The order is driven by sonic logging. MPI moves later because 2.5D borehole
problems fit on one node or GPU; it becomes essential for 3D.

| Milestone | Content | Verified by |
|---|---|---|
| **M0** Foundations | CMake + scikit-build, CI (CPU), GLL/GLJ basis, macro-mesh + forest data model (uniform refinement), leaf geometry and geometry kinds, dof map + deterministic gather/scatter, HDF5 I/O, Python bindings | unit tests |
| **M1** 2D Cartesian core (CPU) | elastic + acoustic (pressure) element blocks, leapfrog, Stacey, free/rigid boundaries, forcings, point receivers, energy monitor, layered template, OpenMP | analytic 2D, Lamb, convergence, energy |
| **M2** Coupling + refinement | fluid–solid coupling, adaptive quadtree + 2:1 balance + hanging nodes, Gmsh import, SEG-Y/XDMF, YAML | Scholte/reflection, refinement-interface reflection |
| **M3** Axisymmetric monopole | axisym blocks (GLJ axis), borehole template (open/cased hole, beds), monopole source, DWM reference solver, STC post-processing | 3D analytic on axis, DWM open and cased hole |
| **M4** Dipole (2.5D, m = 1) | azimuthal-mode blocks, mode-dependent axis conditions, dipole source and receivers, flexural dispersion tools | DWM dipole, flexural dispersion |
| **M5** CUDA backend | all blocks + gather/scatter + observers on GPU, strict-FP mode, batched multi-simulation runs | bitwise CPU/CUDA equivalence, performance |
| **M6** C-PML | PML blocks (all geometry kinds, including axisym and modes), LDDRK | reflection vs Stacey, guided-wave absorption |
| **M7** Local time stepping | Diaz–Grote multi-rate leapfrog, per-level blocks | energy, accuracy, speed-up on cased-hole cases |
| **M8** MPI | SFC partitioning, halo exchange (CPU and CUDA), distributed observers | 1-vs-N-rank bitwise, scaling |
| **M9** 3D | hex blocks, octree forest, deviated wells, 3D microseismic, DAS fibers | 3D analytic, Lamb 3D |
| Later | quadrupole (m = 2), tool body, VTI, attenuation, topography, adjoint/inversion | |

M0–M4 can be developed and fully verified in this environment (CPU). M5 and
M8 need your GPU and MPI hardware for final validation.
