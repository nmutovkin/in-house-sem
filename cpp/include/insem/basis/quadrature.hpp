#pragma once

#include <vector>

namespace insem::basis {

// A 1D quadrature rule on [-1, 1].
struct Rule {
  std::vector<double> points;   // ascending
  std::vector<double> weights;
  int size() const { return static_cast<int>(points.size()); }
};

// Jacobi polynomial P_n^{(a,b)}(x), by the three-term recurrence.
double jacobi(int n, double a, double b, double x);

// d/dx P_n^{(a,b)}(x) = (n + a + b + 1) / 2 * P_{n-1}^{(a+1,b+1)}(x).
double jacobi_derivative(int n, double a, double b, double x);

// The n roots of P_n^{(a,b)} in ascending order (Newton iteration with
// deflation, starting from Chebyshev points).
std::vector<double> jacobi_roots(int n, double a, double b);

// Gauss-Lobatto-Jacobi rule with `n_points` >= 2 points for the weight
// function (1 - x)^alpha (1 + x)^beta, with alpha, beta non-negative integers.
// It contains both end points and integrates polynomials of degree
// 2 * n_points - 3 exactly against the weight.
Rule gauss_lobatto_jacobi(int n_points, int alpha, int beta);

// Gauss-Lobatto-Legendre rule: the SEM collocation points.
inline Rule gll(int n_points) { return gauss_lobatto_jacobi(n_points, 0, 0); }

// Gauss-Lobatto-Jacobi (0,1) rule, weight (1 + x): used in the radial
// direction of axisymmetric elements whose xi = -1 edge lies on the axis.
inline Rule glj01(int n_points) { return gauss_lobatto_jacobi(n_points, 0, 1); }

// Gauss-Legendre rule (interior points only), used for exact integration of
// polynomial moments.
Rule gauss_legendre(int n_points);

}  // namespace insem::basis
