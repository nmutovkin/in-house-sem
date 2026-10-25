#include "insem/basis/lagrange.hpp"

#include <cmath>
#include <utility>

#include "insem/core/error.hpp"

namespace insem::basis {

LagrangeBasis::LagrangeBasis(std::vector<double> nodes) : nodes_(std::move(nodes)) {
  const int n = size();
  INSEM_CHECK(n >= 1, "Lagrange basis needs at least one node");
  bary_.assign(n, 1.0);
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < n; ++k) {
      if (k == j) continue;
      const double d = nodes_[j] - nodes_[k];
      INSEM_CHECK(d != 0.0, "Lagrange nodes must be distinct");
      bary_[j] /= d;
    }
  }
}

int LagrangeBasis::node_index(double x) const {
  for (int j = 0; j < size(); ++j) {
    if (std::abs(x - nodes_[j]) <= 1e-14 * (1.0 + std::abs(nodes_[j]))) return j;
  }
  return -1;
}

void LagrangeBasis::values(double x, double* out) const {
  const int n = size();
  const int at = node_index(x);
  if (at >= 0) {
    for (int j = 0; j < n; ++j) out[j] = (j == at) ? 1.0 : 0.0;
    return;
  }
  double denom = 0.0;
  for (int j = 0; j < n; ++j) {
    out[j] = bary_[j] / (x - nodes_[j]);
    denom += out[j];
  }
  for (int j = 0; j < n; ++j) out[j] /= denom;
}

std::vector<double> LagrangeBasis::values(double x) const {
  std::vector<double> out(size());
  values(x, out.data());
  return out;
}

void LagrangeBasis::derivatives(double x, double* out) const {
  const int n = size();
  const int at = node_index(x);
  if (at >= 0) {
    double diag = 0.0;
    for (int j = 0; j < n; ++j) {
      if (j == at) continue;
      out[j] = (bary_[j] / bary_[at]) / (nodes_[at] - nodes_[j]);
      diag -= out[j];
    }
    out[at] = diag;
    return;
  }
  values(x, out);
  for (int j = 0; j < n; ++j) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) {
      if (k != j) s += 1.0 / (x - nodes_[k]);
    }
    out[j] *= s;
  }
}

std::vector<double> LagrangeBasis::derivatives(double x) const {
  std::vector<double> out(size());
  derivatives(x, out.data());
  return out;
}

std::vector<double> LagrangeBasis::derivative_matrix() const {
  const int n = size();
  std::vector<double> d(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i) derivatives(nodes_[i], &d[static_cast<std::size_t>(i) * n]);
  return d;
}

std::vector<double> LagrangeBasis::interpolation_matrix(
    const std::vector<double>& targets) const {
  const int n = size();
  std::vector<double> m(targets.size() * n);
  for (std::size_t t = 0; t < targets.size(); ++t) values(targets[t], &m[t * n]);
  return m;
}

}  // namespace insem::basis
