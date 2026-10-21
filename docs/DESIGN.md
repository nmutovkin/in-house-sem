# In-house SEM: Architecture & Roadmap

Status: **draft for review**. Nothing here is implemented yet.

A spectral element method (SEM) simulator for wave propagation in coupled
acoustic (fluid) and elastic (solid) media. Its first targets are
**borehole acoustics** and **passive seismic monitoring**: microseismic
events recorded by downhole geophones, hydrophones and DAS fibers.

---

## 1. Decisions so far

| Topic | Decision |
|---|---|
| Language | C++17 core, Python bindings (pybind11) |
| Dimensions | 2D first (Cartesian **and** axisymmetric r–z); code structured so 3D hexahedra slot in later |
| Physics v1 | Isotropic acoustic + isotropic elastic + **fluid–solid coupling** |
| Physics later | Anisotropy (VTI/TTI), attenuation (SLS memory variables), topography, azimuthal Fourier (2.5D) |
| Meshing | In-house structured/layered mesher (conforming interfaces, borehole geometry) + Gmsh `.msh` import; solver works on generic unstructured quads (hexes later) |
| Absorbing BC | Stacey first, C-PML next |
| Parallelism | **Native CUDA + separate CPU (OpenMP) code**, **MPI** domain decomposition |
| Precision | Templated on real type; `float` default, `double` build for verification |
| User interface | Python API is primary; YAML is a thin layer that maps onto the same API |
| Output | HDF5 (traces, DAS, metadata, snapshots) + SEG-Y export + XDMF/VTK for ParaView |
| Units | SI throughout (m, s, kg/m³, Pa) |

---

## 2. Physics and weak forms

### 2.1 Elastic (solid)

$$\rho\,\ddot{\mathbf u} = \nabla\cdot\boldsymbol\sigma + \mathbf f,\qquad
\boldsymbol\sigma = \lambda\,(\nabla\cdot\mathbf u)\,\mathbf I + 2\mu\,\boldsymbol\varepsilon,\qquad
\boldsymbol\varepsilon = \tfrac12(\nabla\mathbf u + \nabla\mathbf u^T)$$

Weak form: $\int \rho\,\mathbf w\cdot\ddot{\mathbf u} = -\int \nabla\mathbf w:\boldsymbol\sigma + \int \mathbf w\cdot\mathbf f + \oint \mathbf w\cdot\mathbf t$.
The free surface is the natural condition (**t** = 0) and needs no code.

The constitutive law is kept behind one point-wise function (`stress_from_strain`).
Anisotropy (a full `C_ij` per GLL point) and attenuation (memory variables)
later replace that function without changing the element kernels.

### 2.2 Acoustic (fluid): potential formulation

We use the displacement potential χ (as in Chaljub & Valette 2004 and
SPECFEM2D), because it makes coupling with the solid cheap and explicit:

$$\mathbf u = \tfrac{1}{\rho}\nabla\chi,\qquad p = -\ddot\chi,\qquad
\tfrac{1}{\kappa}\,\ddot\chi = \nabla\cdot\!\left(\tfrac{1}{\rho}\nabla\chi\right) + s$$

Weak form: $\int \tfrac1\kappa w\ddot\chi = -\int \tfrac1\rho\nabla w\cdot\nabla\chi + \oint w\,(\mathbf n\cdot\mathbf u) + \int w s$.

* Pressure-release surface (p = 0): Dirichlet χ = 0.
* Rigid wall: natural condition (n·u = 0).
* Density may vary inside the fluid (mud, brine and gas layers).

### 2.3 Fluid–solid coupling

The two conditions on the interface Γ are continuity of normal displacement
and traction **t** = −p **n**:

* **Fluid side:** the boundary term above takes $\mathbf n\cdot\mathbf u_{\text{solid}}$.
* **Solid side:** the traction term becomes $\oint \mathbf w\cdot(\ddot\chi\,\mathbf n)$, with **n** pointing out of the solid.

Explicit time stepping order per step (no iteration needed):
1. Newmark predictor for all fields.
2. Fluid: internal forces + coupling term from the predicted solid displacement → solve for χ̈.
3. Solid: internal forces + coupling traction from the **new** χ̈ → solve for ü.
4. Newmark corrector.

The mesh must be **conforming** at Γ, meaning GLL points are shared
geometrically but the fields are separate. Interface edges are found once
during setup and stored as (fluid element, solid element, edge, normal,
weight·jacobian) lists.

### 2.4 Axisymmetric (r–z) formulation, for boreholes

A borehole with a monopole source on its axis is a 3D problem with
rotational symmetry. Solving it on an (r, z) quad mesh gives the true 3D
wavefield (P and S head waves, Stoneley wave, pseudo-Rayleigh modes) at 2D cost.

* Volume element: dΩ = 2π r dr dz, so every integral is weighted by r.
* Elastic unknowns (u_r, u_z) form a P–SV-like system. The strain has an extra hoop
  component ε_θθ = u_r / r. (The torsional u_θ part decouples and is optional.)
* Fluid unknown: χ(r, z) with the same r-weighting.
* **On the axis (r = 0):** in elements touching the axis we use
  **Gauss–Lobatto–Jacobi (0,1)** points in the r-direction instead of GLL
  (Bernardi et al.; Nissen-Meyer et al. 2007, AxiSEM; Bottero et al. 2016,
  SPECFEM2D-axisym). That quadrature absorbs the r weight and removes the 1/r
  singularity. Axis condition: u_r = 0, with u_r/r → ∂u_r/∂r on the axis.
* Element kernels differ from Cartesian 2D only in (a) the extra hoop terms,
  (b) the quadrature weights, and (c) the derivative matrix on axial elements.
  They are a template specialisation of the same kernel, not a separate code path.

**Limitation that matters for microseismic:** pure axisymmetry only allows
sources **on the axis** whose radiation is axisymmetric: explosion, vertical
force, and vertical CLVD or dipole. A general moment-tensor event off the
borehole axis cannot be represented. Options:
* azimuthal Fourier expansion (modes m = 0, ±1, ±2: "2.5D", as in AxiSEM).
  This also gives dipole and quadrupole sonic-logging sources. It is on the roadmap but not in v1.
* full 3D (milestone M7).

In v1, microseismic work is therefore done in **2D Cartesian (P–SV)**.

### 2.5 Later physics (designed for, not built)

* **Attenuation:** standard linear solids with memory variables per GLL point.
  Adds a state array and a point-wise update, and changes no kernel structure.
* **Anisotropy:** `C_ij` per GLL point (VTI/TTI are relevant for shale microseismic).
* **Topography and curved interfaces:** already supported by the geometry
  (quad9 / higher-order mapping). Only the mesher needs extending.

---

## 3. Discretisation

* **Basis:** Lagrange polynomials on Gauss–Lobatto–Legendre (GLL) points,
  degree N (default **N = 4**, so 5×5 points per element). GLJ is used on
  axial elements (§2.4).
* **Quadrature:** collocated GLL, which gives a **diagonal mass matrix**, so explicit time stepping needs no linear solve.
* **Kernels:** tensor-product sum factorisation, with derivative matrices
  `hprime[N+1][N+1]`. N is a **compile-time template parameter**, so loops are fully
  unrolled on both CPU and GPU. We instantiate N = 2…8.
* **Geometry:** isoparametric mapping from 4-node (bilinear) or 9-node
  (quadratic, curved) control points. ∂ξ/∂x and the Jacobian are precomputed
  per GLL point.
* **Assembly:** local→global index map (`ibool[e][i][j]`); assembled fields
  live on unique global points. Material properties are stored **per GLL point**,
  so smooth heterogeneity inside an element is allowed.
* **Resolution rule:** about 5 GLL points per minimum wavelength. For N = 4 that
  means element size h ≲ 0.8 λ_min, where λ_min = v_min / f_max (v_min is the
  slowest wave: S in solids, P in fluids).
* **Time stepping:** explicit Newmark (β = 0, γ = ½), which is second-order
  central difference. Δt = C · min(Δx_GLL / v_p), with C ≈ 0.5 in 2D. The code
  computes Δt automatically and can check a user-supplied value. LDDRK4-6 can be
  added later if C-PML needs it.

**Borehole multiscale issue:** a borehole radius (~0.1 m) is far smaller than
the formation wavelengths at logging frequencies (~0.3–3 m at 1–20 kHz).
The tiny elements near the hole set Δt for the whole model. Mitigations, in order:
radial grading in the mesher (v1), then doubling layers, then local time stepping (future).

---

## 4. Boundaries

| Boundary | Elastic | Acoustic |
|---|---|---|
| Free / pressure-release | natural (t = 0) | Dirichlet χ = 0 |
| Rigid | Dirichlet u = 0 | natural (n·u = 0) |
| Stacey (v1) | t = −ρ[c_p (v·n)n + c_s (v − (v·n)n)] | n·u = −χ̇ / (ρc) |
| C-PML (M6) | ADE/convolutional memory variables in a layer of elements | same |
| Axis (axisym) | u_r = 0 + GLJ quadrature | GLJ quadrature |

Boundaries are stored as edge lists tagged by type. The mesher tags them
automatically, and Gmsh *physical groups* map to the same tags.

---

## 5. Sources and receivers

All are located at **arbitrary positions**, not snapped to GLL points. Each
one is found by an inverse map (Newton iteration in reference coordinates) and
then evaluated with Lagrange interpolation, or with its derivative for
moment tensors and strain.

**Sources**
* Point force (solid), and a **moment tensor** (solid) for microseismic events.
  In 2D Cartesian this uses M_xx, M_zz and M_xz. In axisymmetric mode only the
  on-axis axisymmetric components are allowed (§2.4).
* Pressure / volume-injection monopole (fluid). This is the borehole sonic source.
* Source time functions: Ricker, Gaussian and its derivatives, a smoothed
  Heaviside, or a user-supplied array (resampled to Δt).
* Many simultaneous sources are allowed (multi-event microseismic catalogues,
  noise studies).

**Receivers (point)**
* Solid: displacement, velocity, acceleration. Fluid: pressure (−χ̈), and
  particle displacement or velocity (∇χ/ρ).
* Receivers whose location is on a fluid–solid interface must say which side they record.

**DAS (distributed acoustic sensing)**
* A fiber is a polyline (2D now, 3D curve later) with channel spacing Δs and gauge length L.
* Output: axial strain or strain rate, either point-wise ε_tt = **t**·ε·**t**
  or gauge-averaged [**u**(s+L/2) − **u**(s−L/2)]·**t** / L (the default, which matches
  real interrogators). The averaging is done in the solver at every step, so
  the full wavefield is never stored.
* A fiber inside a fluid-filled borehole is an open modelling question (see §10).

Recording happens on the device (GPU) into a buffer. The buffer is copied
to the host in chunks every K steps.

---

## 6. Meshing

### 6.1 In-house structured mesher (Python, v1)
* **Layered model:** a stack of layers bounded by interfaces z_k(x) (or z_k(r)),
  each layer with its own material and element count. The interfaces are honoured
  exactly, so material and fluid–solid boundaries always fall on element edges.
* **Borehole template (axisymmetric):** fluid column r < a, then optional
  logging-tool body, casing and cement annuli, then formation layers in z.
  Radial grading goes from fine at the borehole wall to coarse in the far
  field, sized by the local λ_min rule.
* Boundaries are tagged automatically (free surface, absorbing, axis, fluid–solid).
* Later additions: doubling bricks, topography, 3D extrusion and hex templates.

### 6.2 Gmsh import
* `.msh` v4 with quad4 and quad9 elements (hex8 and hex27 in 3D later).
  Physical groups map to materials and boundary tags.
* Arbitrary geometry is meshed in Gmsh (its quad recombination is robust in
  2D). We do **not** write a general unstructured quad or hex mesher.

### 6.3 Mesh quality checks
Checks for negative or degenerate Jacobians, skewness, points per wavelength
per element, and the Δt bound per element (which shows what limits the time step).

---

## 7. Software architecture

### 7.1 Data flow: Python front-end, C++ solver

```
 Python (or YAML → Python)                         C++ solver
 ─────────────────────────                         ──────────
 model + mesher / Gmsh ──► Mesh ──► partition ──► simulation database (HDF5)
 sources, receivers, DAS                             │
 run parameters                                      ▼
                                       sem_solver db.h5  (mpirun, CUDA or CPU)
                                                     │
                                                     ▼
                                              results.h5 ──► Python post-proc,
                                                             SEG-Y, XDMF/ParaView
```

* **Large and HPC runs:** Python writes a self-contained HDF5 database, and the
  standalone `sem_solver` executable runs it under `mpirun`. No Python is
  needed on compute nodes.
* **Small runs and notebooks:** the same C++ library is called in-process
  through pybind11 (`sim.run()`). Results come back as NumPy arrays without copying.
* YAML files are parsed in Python only and build the same objects as the API.

### 7.2 Repository layout (proposed)

```
cpp/
  include/sem/
    core/        types.hpp (real_t, index), config, error handling
    basis/       gll.hpp, glj.hpp, lagrange.hpp        (host only)
    mesh/        mesh.hpp, geometry.hpp, numbering.hpp
    physics/     pointwise.hpp  ← shared SEM_HD inline physics (CPU+CUDA)
                 elastic.hpp, acoustic.hpp, coupling.hpp, stacey.hpp
    sources/     source.hpp, stf.hpp
    receivers/   receiver.hpp, das.hpp
    solver/      solver.hpp, newmark.hpp, backend.hpp (interface)
    io/          hdf5_db.hpp, hdf5_out.hpp
    parallel/    mpi_comm.hpp, halo.hpp
  src/
    cpu/         OpenMP kernels
    cuda/        .cu kernels
    ...
  apps/sem_solver.cpp
python/insem/    mesher/, gmsh_io, model, sources, receivers, das, yaml_cfg,
                 database writer, postproc (segy, xdmf), _core (pybind11)
tests/cpp/       Catch2 unit tests
tests/python/    pytest: verification against analytic solutions
examples/        borehole_monopole/, microseismic_2d/, das_vsp/
docs/
```

The package name `insem` is a placeholder (see §10).

### 7.3 Backend design: native CUDA alongside CPU

You chose two code paths. The risk with two paths is drift, so the design
keeps the duplicated part as small as possible:

* **Shared point-wise physics.** Everything that happens at a single GLL point
  (strain → stress, Stacey traction, coupling term, source and receiver
  interpolation weights, DAS projection) lives in `physics/pointwise.hpp` as
  `SEM_HD inline` functions (`__host__ __device__` under nvcc, plain `inline`
  otherwise). Both backends call the same functions.
* **Duplicated part:** only the loop and thread mapping, and the scatter-add.
  * CPU: OpenMP over **coloured** element sets, so the global assembly needs no atomics. SIMD comes from the compile-time N.
  * CUDA: one thread per GLL point, several elements per block (5×5 = 25
    points is too small for one block). Derivative matrices are in shared or
    constant memory, and assembly uses `atomicAdd`, with colouring available as a fallback.
* **One data layout** for both: structure-of-arrays, element-contiguous
  `[nspec][N+1][N+1]` for geometry and material, and `[nglob]` per component
  for fields. The CPU layout is exactly what gets uploaded to the GPU.
* `Backend` interface: `compute_forces_acoustic`, `compute_forces_elastic`,
  `couple_fluid_solid`, `apply_stacey`, `add_sources`, `record`,
  `update_newmark`, `pack_halo` / `unpack_halo`. `CpuBackend` and `CudaBackend` implement it.
* **Equivalence tests:** every physics test runs on both backends, and the
  CUDA results must match the CPU results within a float tolerance. These tests
  run on your GPU machines; this development container has no GPU or `nvcc`.

### 7.4 MPI

* Partitioning happens in Python before the run. Options are METIS (via
  pymetis) and a simple geometric fallback, which keeps C++ dependencies small.
* Each rank owns its elements. Global points shared between ranks are listed
  per neighbour, and their contributions are **summed** across ranks after
  each force computation (classic SEM assembly).
* Communication overlaps with computation: outer elements → post non-blocking
  send/recv → inner elements → wait → unpack.
* GPU halos are packed on the device. CUDA-aware MPI is used when available,
  otherwise the halos are staged through pinned host buffers.
* Fluid–solid interfaces and receivers may cross partition boundaries. Each
  receiver is owned by exactly one rank.

### 7.5 Build and dependencies

* CMake ≥ 3.24 with options `SEM_ENABLE_CUDA`, `SEM_ENABLE_MPI`,
  `SEM_ENABLE_OPENMP` and `SEM_REAL=float|double`, plus a list of N values to instantiate.
* Required: HDF5 (C API; parallel HDF5 when MPI is on).
* Optional: CUDA ≥ 12, MPI.
* Python: pybind11, numpy, h5py, pyyaml, optional gmsh and pymetis. Built with scikit-build-core.
* Tests: Catch2 (C++) and pytest (Python). CI here covers the CPU builds only.

---

## 8. Output

HDF5 `results.h5`:
```
/meta               run parameters, dt, nt, units, code version, git hash
/receivers/<net.sta>/{displacement|velocity|acceleration|pressure}  [nt, ncomp]
/receivers/<net.sta>@{position, medium, component names}
/das/<fiber>/{strain|strain_rate} [nt, nchannels]
/das/<fiber>@{gauge_length, channel_positions, tangents}
/snapshots/<field>/<step>  (optional, decimated in time and space)
```
* SEG-Y rev 2 export for receiver gathers and DAS (one trace per channel).
* XDMF sidecar files over the HDF5 snapshots, so ParaView opens them directly.

---

## 9. Verification plan

Physics is not considered done until it passes these tests.

| Test | Reference |
|---|---|
| GLL/GLJ points, weights, derivative matrices | closed forms / high-precision tables |
| 2D homogeneous acoustic, point source | analytic 2D Green's function convolved with the STF |
| 2D homogeneous elastic, point force / moment tensor | analytic 2D elastic Green's function |
| Lamb's problem (free surface, Rayleigh wave) | Garvin / Lamb analytic |
| Fluid–solid interface (reflection, Scholte wave) | Gar6more2D (open-source analytic code) |
| Spectral convergence: error vs N and h | expected exponential and algebraic rates |
| Energy conservation (closed box, rigid/free) | total energy constant to round-off |
| Stacey reflection strength vs incidence angle | measured against a larger reference domain |
| Axisym homogeneous fluid / solid point source on axis | analytic 3D spherical-wave solutions |
| Borehole monopole (Stoneley, head waves) | discrete-wavenumber method (DWM) reference in Python (Tsang & Rader 1979) |
| DAS gauge-length response | strain computed from analytic displacement |
| CUDA vs CPU, 1 rank vs N ranks | bitwise or tolerance equivalence |

---

## 10. Open questions for review

1. **Name.** The package and namespace name (`insem` is a placeholder).
2. **Typical problem scales.** Frequencies (sonic logging is about 1–20 kHz;
   microseismic is about 10–500 Hz), domain sizes, and run lengths. These set
   memory and compute targets and tell us how urgent MPI is for 2D.
3. **Borehole geometry detail.** Do you need the logging-tool body, casing and
   cement in v1, or just a fluid column in the formation?
4. **DAS in boreholes.** Is the fiber cemented behind casing (it records the
   solid's strain) or in the fluid? The default proposal is to attach the
   fiber to the solid medium it sits in.
5. **Microseismic in 2D.** Is 2D Cartesian P–SV acceptable for v1 microseismic
   studies, or should azimuthal Fourier (2.5D) or 3D move earlier in the roadmap?
6. **Target hardware.** GPU models or compute capability, MPI implementation,
   and whether CUDA-aware MPI is available. Is there any GPU CI runner?
7. **Licensing and dependencies.** Is any open-source licence acceptable for
   dependencies (all proposed ones are BSD, MIT or Apache)?
8. **Gmsh at runtime.** Is the Gmsh Python package acceptable, or should we read `.msh` files only?

---

## 11. Roadmap

| Milestone | Content | Verified by |
|---|---|---|
| **M0** Skeleton | CMake, scikit-build packaging, CI (CPU), GLL/GLJ basis, mesh data structures, HDF5 database I/O | unit tests |
| **M1** 2D Cartesian core (CPU) | acoustic + elastic, structured layered mesher, Stacey, free/rigid boundaries, point force / MT / pressure sources, point receivers, HDF5 output, OpenMP | analytic 2D, Lamb, convergence, energy |
| **M2** Coupling + I/O | fluid–solid coupling, Gmsh import, DAS fibers, SEG-Y/XDMF export, YAML front-end | Gar6more2D, DAS analytic |
| **M3** Axisymmetric | GLJ axis elements, axisym acoustic, elastic and coupling, borehole mesher with radial grading, monopole source | 3D analytic on axis, DWM borehole |
| **M4** CUDA backend | all M1–M3 physics on GPU, device-side recording | CPU/GPU equivalence + performance |
| **M5** MPI | partitioning, halo exchange (CPU and CUDA, CUDA-aware), distributed recording | 1-vs-N-rank equivalence, scaling |
| **M6** C-PML | elastic + acoustic + coupled corners | reflection vs Stacey |
| **M7** 3D | hex elements, 3D mesher/extrusion, Gmsh hex import, 3D microseismic with borehole arrays and 3D fibers | 3D analytic, Lamb 3D |
| Later | attenuation, anisotropy (VTI/TTI), topography, 2.5D azimuthal Fourier (dipole logging), doubling layers, local time stepping, adjoint (microseismic location and inversion) | |

M1–M3 are mostly CPU work and can be developed and tested fully in this
environment. M4 and M5 need your GPU and MPI hardware for final validation.
