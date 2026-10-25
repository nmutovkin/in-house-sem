#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "insem/basis/lagrange.hpp"
#include "insem/basis/quadrature.hpp"

using Catch::Matchers::WithinAbs;
using namespace insem::basis;

namespace {

// A polynomial of degree `deg` and its derivative.
double poly(int deg, double x) {
  double v = 0.0;
  for (int k = 0; k <= deg; ++k) v += (k + 1) * 0.3 * std::pow(x, k);
  return v;
}
double dpoly(int deg, double x) {
  double v = 0.0;
  for (int k = 1; k <= deg; ++k) v += (k + 1) * 0.3 * k * std::pow(x, k - 1);
  return v;
}

}  // namespace

TEST_CASE("Lagrange values form a partition of unity and are nodal", "[basis]") {
  const LagrangeBasis b(gll(6).points);
  for (int i = 0; i < b.size(); ++i) {
    const auto v = b.values(b.nodes()[i]);
    for (int j = 0; j < b.size(); ++j) CHECK(v[j] == (i == j ? 1.0 : 0.0));
  }
  for (double x : {-0.95, -0.2, 0.33, 0.999}) {
    const auto v = b.values(x);
    double s = 0.0;
    for (double vj : v) s += vj;
    CHECK_THAT(s, WithinAbs(1.0, 1e-14));
  }
}

TEST_CASE("Lagrange interpolation and derivatives are exact for degree <= N", "[basis]") {
  const int n = GENERATE(2, 3, 5, 7, 9);
  for (const auto& nodes : {gll(n).points, glj01(n).points}) {
    const LagrangeBasis b(nodes);
    const int deg = n - 1;
    std::vector<double> f(n);
    for (int j = 0; j < n; ++j) f[j] = poly(deg, nodes[j]);

    // Derivative matrix at the nodes.
    const auto d = b.derivative_matrix();
    for (int i = 0; i < n; ++i) {
      double df = 0.0;
      for (int j = 0; j < n; ++j) df += d[i * n + j] * f[j];
      CHECK_THAT(df, WithinAbs(dpoly(deg, nodes[i]), 1e-11));
    }
    // Values and derivatives away from the nodes.
    for (double x : {-0.77, 0.123, 0.5}) {
      const auto v = b.values(x);
      const auto dv = b.derivatives(x);
      double fx = 0.0, dfx = 0.0;
      for (int j = 0; j < n; ++j) {
        fx += v[j] * f[j];
        dfx += dv[j] * f[j];
      }
      CHECK_THAT(fx, WithinAbs(poly(deg, x), 1e-12));
      CHECK_THAT(dfx, WithinAbs(dpoly(deg, x), 1e-10));
    }
  }
}

TEST_CASE("Derivative matrix rows sum to zero", "[basis]") {
  const LagrangeBasis b(gll(8).points);
  const auto d = b.derivative_matrix();
  for (int i = 0; i < 8; ++i) {
    double s = 0.0;
    for (int j = 0; j < 8; ++j) s += d[i * 8 + j];
    CHECK_THAT(s, WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("Interpolation matrix matches values", "[basis]") {
  const LagrangeBasis b(gll(4).points);
  const std::vector<double> targets{-1.0, -0.1, 0.4};
  const auto m = b.interpolation_matrix(targets);
  for (std::size_t t = 0; t < targets.size(); ++t) {
    const auto v = b.values(targets[t]);
    for (int j = 0; j < 4; ++j) CHECK(m[t * 4 + j] == v[j]);
  }
}
