#include "insem/sim/description.hpp"

#include <cmath>
#include <utility>

#include "insem/core/error.hpp"

namespace insem {
namespace {

constexpr double kPi = 3.14159265358979323846;

double default_t0(double f0, double t0) { return t0 < 0.0 ? 1.2 / f0 : t0; }

}  // namespace

std::string to_string(Medium m) { return m == Medium::Acoustic ? "acoustic" : "elastic"; }

Medium medium_from_string(const std::string& name) {
  if (name == "acoustic") return Medium::Acoustic;
  if (name == "elastic") return Medium::Elastic;
  throw Error("unknown medium '" + name + "' (expected 'acoustic' or 'elastic')");
}

void Material::validate() const {
  INSEM_CHECK(rho > 0.0 && std::isfinite(rho), "density must be positive, got " << rho);
  INSEM_CHECK(vp > 0.0 && std::isfinite(vp), "P velocity must be positive, got " << vp);
  if (medium == Medium::Elastic) {
    INSEM_CHECK(vs > 0.0, "elastic S velocity must be positive, got " << vs
                                                                       << " (use an acoustic material for fluids)");
    // Plane-strain stability needs lambda + mu = rho (vp^2 - vs^2) > 0.
    INSEM_CHECK(vp > vs, "elastic P velocity (" << vp << ") must exceed S velocity (" << vs << ")");
  }
}

std::string to_string(BoundaryType b) {
  switch (b) {
    case BoundaryType::Free:
      return "free";
    case BoundaryType::Rigid:
      return "rigid";
    case BoundaryType::Absorbing:
      return "absorbing";
  }
  return "unknown";
}

BoundaryType boundary_type_from_string(const std::string& name) {
  if (name == "free") return BoundaryType::Free;
  if (name == "rigid") return BoundaryType::Rigid;
  if (name == "absorbing") return BoundaryType::Absorbing;
  throw Error("unknown boundary type '" + name + "' (expected 'free', 'rigid' or 'absorbing')");
}

SourceTimeFunction SourceTimeFunction::ricker(double f0, double t0, double amplitude) {
  INSEM_CHECK(f0 > 0.0, "Ricker frequency must be positive");
  SourceTimeFunction s;
  s.kind = Kind::Ricker;
  s.f0 = f0;
  s.t0 = default_t0(f0, t0);
  s.amplitude = amplitude;
  return s;
}

SourceTimeFunction SourceTimeFunction::gaussian(double f0, double t0, double amplitude) {
  SourceTimeFunction s = ricker(f0, t0, amplitude);
  s.kind = Kind::Gaussian;
  return s;
}

SourceTimeFunction SourceTimeFunction::gaussian_derivative(double f0, double t0, double amplitude) {
  SourceTimeFunction s = ricker(f0, t0, amplitude);
  s.kind = Kind::GaussianDerivative;
  return s;
}

SourceTimeFunction SourceTimeFunction::sampled(std::vector<double> samples, double dt, double amplitude) {
  INSEM_CHECK(dt > 0.0, "sample interval must be positive");
  INSEM_CHECK(!samples.empty(), "sampled source time function needs samples");
  SourceTimeFunction s;
  s.kind = Kind::Samples;
  s.samples = std::move(samples);
  s.sample_dt = dt;
  s.amplitude = amplitude;
  return s;
}

double SourceTimeFunction::operator()(double t) const {
  switch (kind) {
    case Kind::Ricker: {
      const double a = std::pow(kPi * f0 * (t - t0), 2);
      return amplitude * (1.0 - 2.0 * a) * std::exp(-a);
    }
    case Kind::Gaussian: {
      const double a = std::pow(kPi * f0 * (t - t0), 2);
      return amplitude * std::exp(-a);
    }
    case Kind::GaussianDerivative: {
      const double tau = t - t0;
      const double c = std::pow(kPi * f0, 2);
      return amplitude * (-2.0 * c * tau) * std::exp(-c * tau * tau);
    }
    case Kind::Samples: {
      const double x = t / sample_dt;
      if (x < 0.0 || x > static_cast<double>(samples.size() - 1)) return 0.0;
      const auto i = static_cast<std::size_t>(x);
      if (i + 1 >= samples.size()) return amplitude * samples.back();
      const double w = x - static_cast<double>(i);
      return amplitude * ((1.0 - w) * samples[i] + w * samples[i + 1]);
    }
  }
  return 0.0;
}

std::string SourceTimeFunction::kind_name() const {
  switch (kind) {
    case Kind::Ricker:
      return "ricker";
    case Kind::Gaussian:
      return "gaussian";
    case Kind::GaussianDerivative:
      return "gaussian_derivative";
    case Kind::Samples:
      return "samples";
  }
  return "unknown";
}

Source Source::force(double x, double z, double fx, double fz, SourceTimeFunction stf) {
  return {Kind::Force, {x, z}, {fx, fz, 0.0}, std::move(stf)};
}

Source Source::moment_tensor(double x, double z, double mxx, double mzz, double mxz,
                             SourceTimeFunction stf) {
  return {Kind::MomentTensor, {x, z}, {mxx, mzz, mxz}, std::move(stf)};
}

Source Source::monopole(double x, double z, SourceTimeFunction stf) {
  return {Kind::Monopole, {x, z}, {1.0, 0.0, 0.0}, std::move(stf)};
}

std::string Source::kind_name() const {
  switch (kind) {
    case Kind::Force:
      return "force";
    case Kind::MomentTensor:
      return "moment_tensor";
    case Kind::Monopole:
      return "monopole";
  }
  return "unknown";
}

int quantity_components(const std::string& quantity, Medium medium) {
  if (medium == Medium::Elastic) {
    if (quantity == "displacement" || quantity == "velocity" || quantity == "acceleration") return 2;
  } else {
    if (quantity == "pressure") return 1;
    if (quantity == "particle_velocity") return 2;
  }
  throw Error("quantity '" + quantity + "' cannot be recorded in " + to_string(medium) +
              " media (elastic: displacement, velocity, acceleration; acoustic: pressure, "
              "particle_velocity)");
}

}  // namespace insem
