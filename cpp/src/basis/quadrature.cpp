#include "insem/basis/quadrature.hpp"

#include <cmath>
#include <utility>

#include "insem/core/error.hpp"

namespace insem::basis {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Solves the dense n x n system A x = b (row-major) by Gaussian elimination
// with partial pivoting. Only used for small quadrature moment systems.
std::vector<double> solve_dense(std::vector<double> a, std::vector<double> b,
                                int n) {
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    for (int row = col + 1; row < n; ++row) {
      if (std::abs(a[row * n + col]) > std::abs(a[pivot * n + col])) {
        pivot = row;
      }
    }
    INSEM_CHECK(a[pivot * n + col] != 0.0, "singular quadrature moment system");
    if (pivot != col) {
      for (int k = 0; k < n; ++k) std::swap(a[col * n + k], a[pivot * n + k]);
      std::swap(b[col], b[pivot]);
    }
    for (int row = col + 1; row < n; ++row) {
      const double f = a[row * n + col] / a[col * n + col];
      for (int k = col; k < n; ++k) a[row * n + k] -= f * a[col * n + k];
      b[row] -= f * b[col];
    }
  }
  std::vector<double> x(n);
  for (int row = n - 1; row >= 0; --row) {
    double s = b[row];
    for (int k = row + 1; k < n; ++k) s -= a[row * n + k] * x[k];
    x[row] = s / a[row * n + row];
  }
  return x;
}

}  // namespace

double jacobi(int n, double a, double b, double x) {
  INSEM_CHECK(n >= 0, "Jacobi polynomial degree must be non-negative");
  if (n == 0) return 1.0;
  double p_prev = 1.0;
  double p = 0.5 * ((a + b + 2.0) * x + (a - b));
  for (int k = 1; k < n; ++k) {
    const double c = 2.0 * k + a + b;
    const double a1 = 2.0 * (k + 1) * (k + a + b + 1) * c;
    const double a2 = (c + 1) * (a * a - b * b);
    const double a3 = (c + 1) * (c + 2) * c;
    const double a4 = 2.0 * (k + a) * (k + b) * (c + 2);
    const double p_next = ((a2 + a3 * x) * p - a4 * p_prev) / a1;
    p_prev = p;
    p = p_next;
  }
  return p;
}

double jacobi_derivative(int n, double a, double b, double x) {
  if (n == 0) return 0.0;
  return 0.5 * (n + a + b + 1) * jacobi(n - 1, a + 1, b + 1, x);
}

std::vector<double> jacobi_roots(int n, double a, double b) {
  INSEM_CHECK(n >= 0, "number of roots must be non-negative");
  std::vector<double> roots(n);
  for (int k = 0; k < n; ++k) {
    double r = -std::cos((2.0 * k + 1.0) * kPi / (2.0 * n));
    if (k > 0) r = 0.5 * (r + roots[k - 1]);
    for (int it = 0; it < 200; ++it) {
      double deflation = 0.0;
      for (int i = 0; i < k; ++i) deflation += 1.0 / (r - roots[i]);
      const double p = jacobi(n, a, b, r);
      const double delta = -p / (jacobi_derivative(n, a, b, r) - deflation * p);
      r += delta;
      if (std::abs(delta) < 1e-16) break;
    }
    roots[k] = r;
  }
  return roots;
}

Rule gauss_legendre(int n_points) {
  INSEM_CHECK(n_points >= 1, "Gauss-Legendre rule needs at least one point");
  Rule rule;
  rule.points = jacobi_roots(n_points, 0.0, 0.0);
  rule.weights.resize(n_points);
  for (int i = 0; i < n_points; ++i) {
    const double x = rule.points[i];
    const double dp = jacobi_derivative(n_points, 0.0, 0.0, x);
    rule.weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
  }
  return rule;
}

Rule gauss_lobatto_jacobi(int n_points, int alpha, int beta) {
  INSEM_CHECK(n_points >= 2, "Gauss-Lobatto rule needs at least two points");
  INSEM_CHECK(alpha >= 0 && beta >= 0,
              "Gauss-Lobatto-Jacobi weight exponents must be non-negative integers");

  Rule rule;
  rule.points.reserve(n_points);
  rule.points.push_back(-1.0);
  for (double r : jacobi_roots(n_points - 2, alpha + 1.0, beta + 1.0)) {
    rule.points.push_back(r);
  }
  rule.points.push_back(1.0);

  // Interpolatory weights: sum_j w_j P_k(x_j) = int (1-x)^a (1+x)^b P_k(x) dx
  // for the Legendre polynomials P_k, k < n_points. The moments are exact
  // integrals of polynomials, computed with a large enough Gauss-Legendre rule.
  const Rule exact = gauss_legendre(n_points + alpha + beta + 2);
  std::vector<double> moments(n_points, 0.0);
  for (int q = 0; q < exact.size(); ++q) {
    const double x = exact.points[q];
    const double w = exact.weights[q] * std::pow(1.0 - x, alpha) * std::pow(1.0 + x, beta);
    for (int k = 0; k < n_points; ++k) moments[k] += w * jacobi(k, 0.0, 0.0, x);
  }
  std::vector<double> vandermonde(static_cast<std::size_t>(n_points) * n_points);
  for (int k = 0; k < n_points; ++k) {
    for (int j = 0; j < n_points; ++j) {
      vandermonde[k * n_points + j] = jacobi(k, 0.0, 0.0, rule.points[j]);
    }
  }
  rule.weights = solve_dense(std::move(vandermonde), std::move(moments), n_points);
  return rule;
}

}  // namespace insem::basis
