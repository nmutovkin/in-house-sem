#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace insem::io {

// Thin RAII wrapper over the HDF5 C API covering what case and result files
// need: n-dimensional datasets of double, float, int32 and int64, groups, and
// scalar attributes. Paths are absolute ("/mesh/nodes"); missing parent
// groups are created on write, and existing datasets/attributes are replaced.
class H5File {
 public:
  enum class Mode { Read, ReadWrite, Truncate };

  H5File(const std::string& path, Mode mode);
  ~H5File();
  H5File(const H5File&) = delete;
  H5File& operator=(const H5File&) = delete;
  H5File(H5File&& other) noexcept;
  H5File& operator=(H5File&& other) noexcept;

  const std::string& path() const { return path_; }

  bool exists(const std::string& path) const;
  void create_group(const std::string& path);

  template <class T>
  void write(const std::string& path, const T* data, const std::vector<std::size_t>& dims);
  template <class T>
  void write(const std::string& path, const std::vector<T>& data) {
    write(path, data.data(), {data.size()});
  }

  // Reads a whole dataset; `dims` receives its shape when non-null.
  template <class T>
  std::vector<T> read(const std::string& path, std::vector<std::size_t>* dims = nullptr) const;

  void set_attribute(const std::string& object, const std::string& name, const std::string& value);
  void set_attribute(const std::string& object, const std::string& name, std::int64_t value);
  void set_attribute(const std::string& object, const std::string& name, double value);
  bool has_attribute(const std::string& object, const std::string& name) const;
  std::string attribute_string(const std::string& object, const std::string& name) const;
  std::int64_t attribute_int(const std::string& object, const std::string& name) const;
  double attribute_double(const std::string& object, const std::string& name) const;

 private:
  void close() noexcept;

  std::int64_t id_ = -1;  // hid_t
  std::string path_;
};

}  // namespace insem::io
