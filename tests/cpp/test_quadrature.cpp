#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numeric>

#include "insem/basis/quadrature.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace insem::basis;

TEST_CASE("Legendre polynomials match closed forms", "[basis]") {
  for (double x : {-1.0, -0.3, 0.0, 0.7, 1.0}) {
    CHECK_THAT(jacobi(2, 0, 0, x), WithinAbs(0.5 * (3 * x * x - 1), 1e-15));
    CHECK_THAT(jacobi(3, 0, 0, x), WithinAbs(0.5 * (5 * x * x * x - 3 * x), 1e-15));
    CHECK_THAT(jacobi_derivative(3, 0, 0, x), WithinAbs(0.5 * (15 * x * x - 3), 1e-14));
  }
}

TEST_CASE("GLL rule with 5 points matches the closed form", "[basis]") {
  const Rule r = gll(5);
  const double a = std::sqrt(3.0 / 7.0);
  const double expected_x[5] = {-1.0, -a, 0.0, a, 1.0};
  const double expected_w[5] = {0.1, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 0.1};
  for (int i = 0; i < 5; ++i) {
    CHECK_THAT(r.points[i], WithinAbs(expected_x[i], 1e-15));
    CHECK_THAT(r.weights[i], WithinAbs(expected_w[i], 1e-14));
  }
}

TEST_CASE("GLL rules are symmetric and exact to degree 2n-3", "[basis]") {
  const int n = GENERATE(2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16);
  const Rule r = gll(n);
  REQUIRE(r.size() == n);
  for (int i = 0; i < n; ++i) {
    CHECK_THAT(r.points[i], WithinAbs(-r.points[n - 1 - i], 1e-14));
    CHECK_THAT(r.weights[i], WithinAbs(r.weights[n - 1 - i], 1e-13));
    CHECK(r.weights[i] > 0.0);
    if (i > 0) CHECK(r.points[i] > r.points[i - 1]);
  }
  for (int k = 0; k <= 2 * n - 3; ++k) {
    double q = 0.0;
    for (int i = 0; i < n; ++i) q += r.weights[i] * std::pow(r.points[i], k);
    const double exact = (k % 2 == 0) ? 2.0 / (k + 1) : 0.0;
    CHECK_THAT(q, WithinAbs(exact, 1e-13));
  }
  // The usual closed form of the GLL weights.
  for (int i = 0; i < n; ++i) {
    const double p = jacobi(n - 1, 0, 0, r.points[i]);
    CHECK_THAT(r.weights[i], WithinRel(2.0 / ((n - 1) * n * p * p), 1e-12));
  }
}

TEST_CASE("GLJ(0,1) rules integrate (1+x) x^k exactly to degree 2n-3", "[basis]") {
  const int n = GENERATE(2, 3, 4, 5, 6, 8, 10);
  const Rule r = glj01(n);
  REQUIRE(r.size() == n);
  CHECK(r.points.front() == -1.0);
  CHECK(r.points.back() == 1.0);
  for (int k = 0; k <= 2 * n - 3; ++k) {
    double q = 0.0;
    for (int i = 0; i < n; ++i) q += r.weights[i] * std::pow(r.points[i], k);
    // int_{-1}^{1} (1 + x) x^k dx
    const double even = (k % 2 == 0) ? 2.0 / (k + 1) : 0.0;
    const double odd = (k % 2 == 1) ? 2.0 / (k + 2) : 0.0;
    CHECK_THAT(q, WithinAbs(even + odd, 1e-13));
  }
  for (double w : r.weights) CHECK(w >= 0.0);
  // The weight function vanishes at x = -1 only as a factor; the end weight
  // stays positive (this is what makes the axis well behaved).
  CHECK(r.weights.front() > 0.0);
}

TEST_CASE("Gauss-Legendre rules are exact to degree 2n-1", "[basis]") {
  const int n = GENERATE(1, 2, 3, 5, 8, 12);
  const Rule r = gauss_legendre(n);
  for (int k = 0; k <= 2 * n - 1; ++k) {
    double q = 0.0;
    for (int i = 0; i < n; ++i) q += r.weights[i] * std::pow(r.points[i], k);
    CHECK_THAT(q, WithinAbs((k % 2 == 0) ? 2.0 / (k + 1) : 0.0, 1e-13));
  }
}
