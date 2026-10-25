#pragma once

#include <vector>

namespace insem::basis {

// Lagrange polynomials on a set of distinct 1D nodes, evaluated with the
// barycentric formula.
class LagrangeBasis {
 public:
  explicit LagrangeBasis(std::vector<double> nodes);

  int size() const { return static_cast<int>(nodes_.size()); }
  const std::vector<double>& nodes() const { return nodes_; }

  // out[j] = l_j(x), for j < size().
  void values(double x, double* out) const;
  std::vector<double> values(double x) const;

  // out[j] = l_j'(x), for j < size().
  void derivatives(double x, double* out) const;
  std::vector<double> derivatives(double x) const;

  // D[i * n + j] = l_j'(x_i): applying D to nodal values gives nodal
  // values of the derivative.
  std::vector<double> derivative_matrix() const;

  // I[t * n + j] = l_j(targets[t]).
  std::vector<double> interpolation_matrix(const std::vector<double>& targets) const;

 private:
  // Index of the node x coincides with, or -1.
  int node_index(double x) const;

  std::vector<double> nodes_;
  std::vector<double> bary_;
};

}  // namespace insem::basis
