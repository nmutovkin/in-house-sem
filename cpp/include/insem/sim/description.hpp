#pragma once

#include <array>
#include <string>
#include <vector>

namespace insem {

// ---------------------------------------------------------------- materials

enum class Medium : int { Acoustic = 0, Elastic = 1 };

std::string to_string(Medium m);
Medium medium_from_string(const std::string& name);

// Isotropic material of one mesh region (SI units). vs is ignored for acoustic media.
struct Material {
  Medium medium = Medium::Elastic;
  double vp = 0.0;
  double vs = 0.0;
  double rho = 0.0;

  static Material acoustic(double vp, double rho) { return {Medium::Acoustic, vp, 0.0, rho}; }
  static Material elastic(double vp, double vs, double rho) { return {Medium::Elastic, vp, vs, rho}; }

  // Throws insem::Error unless the parameters are physical.
  void validate() const;
};

// ------------------------------------------------------- boundary conditions

// Meaning per medium (docs/DESIGN.md §4):
//   Free:      elastic traction-free (natural); acoustic pressure-release (p = 0)
//   Rigid:     elastic u = 0;                    acoustic rigid wall (natural)
//   Absorbing: Stacey first-order absorbing condition
// Untagged boundary faces behave as the natural condition of their medium
// (free for elastic, rigid for acoustic).
enum class BoundaryType : int { Free = 0, Rigid = 1, Absorbing = 2 };

std::string to_string(BoundaryType b);
BoundaryType boundary_type_from_string(const std::string& name);

// -------------------------------------------------- source time functions

struct SourceTimeFunction {
  enum class Kind : int { Ricker = 0, Gaussian = 1, GaussianDerivative = 2, Samples = 3 };

  Kind kind = Kind::Ricker;
  double f0 = 0.0;         // dominant / characteristic frequency (Hz)
  double t0 = 0.0;         // time shift (s)
  double amplitude = 1.0;
  std::vector<double> samples;  // Kind::Samples: values at t = i * sample_dt
  double sample_dt = 0.0;

  // Ricker wavelet (1 - 2 a) exp(-a), a = (pi f0 (t - t0))^2. Default t0 = 1.2 / f0.
  static SourceTimeFunction ricker(double f0, double t0 = -1.0, double amplitude = 1.0);
  // exp(-a): unit peak Gaussian with the same width parameter as the Ricker.
  static SourceTimeFunction gaussian(double f0, double t0 = -1.0, double amplitude = 1.0);
  // Time derivative of the Gaussian above.
  static SourceTimeFunction gaussian_derivative(double f0, double t0 = -1.0, double amplitude = 1.0);
  // Linearly interpolated samples; zero outside [0, (n - 1) dt].
  static SourceTimeFunction sampled(std::vector<double> samples, double dt, double amplitude = 1.0);

  double operator()(double t) const;
  std::string kind_name() const;
};

// ------------------------------------------------------------------ sources

// Point sources at arbitrary positions (docs/DESIGN.md §5).
//   Force:        components = (fx, fz, -)   in N (per unit length in 2D)
//   MomentTensor: components = (mxx, mzz, mxz) in N m
//   Monopole:     acoustic volume-injection source; the time function is
//                 d^2 Q / dt^2, with Q the injected volume (components unused)
struct Source {
  enum class Kind : int { Force = 0, MomentTensor = 1, Monopole = 2 };

  Kind kind = Kind::Force;
  std::array<double, 2> position{0.0, 0.0};
  std::array<double, 3> components{0.0, 0.0, 0.0};
  SourceTimeFunction stf;

  static Source force(double x, double z, double fx, double fz, SourceTimeFunction stf);
  static Source moment_tensor(double x, double z, double mxx, double mzz, double mxz,
                              SourceTimeFunction stf);
  static Source monopole(double x, double z, SourceTimeFunction stf);

  // The medium the source must lie in.
  Medium medium() const { return kind == Kind::Monopole ? Medium::Acoustic : Medium::Elastic; }
  std::string kind_name() const;
};

// ---------------------------------------------------------------- receivers

// Recorded quantities:
//   elastic:  "displacement", "velocity", "acceleration"   (2 components: x, z)
//   acoustic: "pressure" (1 component), "particle_velocity" (2 components)
struct Receiver {
  std::string name;
  std::array<double, 2> position{0.0, 0.0};
  std::vector<std::string> quantities;
};

// Number of components of a recorded quantity; throws for unknown names or
// quantities that do not exist in `medium`.
int quantity_components(const std::string& quantity, Medium medium);

// ---------------------------------------------------------------- time

struct TimeConfig {
  double dt = 0.0;          // 0: automatic, safety * stable time step
  double safety = 0.8;      // fraction of the estimated stability limit used when dt = 0
  int n_steps = 0;          // number of steps; if 0, derived from duration
  double duration = 0.0;    // simulated time (s), used when n_steps = 0
  int record_every = 1;     // receiver sampling interval in steps
  int energy_every = 0;     // energy monitor interval in steps (0 = off)
};

}  // namespace insem
