#include "insem/io/hdf5.hpp"

#include <hdf5.h>

#include <utility>

#include "insem/core/error.hpp"

namespace insem::io {
namespace {

template <class T>
hid_t native_type();
template <>
hid_t native_type<double>() { return H5T_NATIVE_DOUBLE; }
template <>
hid_t native_type<float>() { return H5T_NATIVE_FLOAT; }
template <>
hid_t native_type<std::int32_t>() { return H5T_NATIVE_INT32; }
template <>
hid_t native_type<std::int64_t>() { return H5T_NATIVE_INT64; }

// Closes an HDF5 handle when it goes out of scope.
class Handle {
 public:
  Handle(hid_t id, herr_t (*closer)(hid_t)) : id_(id), closer_(closer) {}
  ~Handle() {
    if (id_ >= 0) closer_(id_);
  }
  Handle(Handle&& other) noexcept : id_(other.id_), closer_(other.closer_) { other.id_ = -1; }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle& operator=(Handle&&) = delete;
  hid_t get() const { return id_; }

 private:
  hid_t id_;
  herr_t (*closer_)(hid_t);
};

void silence_hdf5_errors() {
  static const bool done = [] {
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    return true;
  }();
  (void)done;
}

// Link-creation property list that creates missing parent groups.
Handle intermediate_groups() {
  Handle lcpl(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
  INSEM_CHECK(lcpl.get() >= 0, "H5Pcreate failed");
  H5Pset_create_intermediate_group(lcpl.get(), 1);
  return lcpl;
}

}  // namespace

H5File::H5File(const std::string& path, Mode mode) : path_(path) {
  silence_hdf5_errors();
  hid_t id = -1;
  switch (mode) {
    case Mode::Read:
      id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      break;
    case Mode::ReadWrite:
      id = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      break;
    case Mode::Truncate:
      id = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      break;
  }
  INSEM_CHECK(id >= 0, "cannot open HDF5 file '" << path << "'");
  id_ = id;
}

H5File::~H5File() { close(); }

H5File::H5File(H5File&& other) noexcept : id_(other.id_), path_(std::move(other.path_)) {
  other.id_ = -1;
}

H5File& H5File::operator=(H5File&& other) noexcept {
  if (this != &other) {
    close();
    id_ = other.id_;
    path_ = std::move(other.path_);
    other.id_ = -1;
  }
  return *this;
}

void H5File::close() noexcept {
  if (id_ >= 0) H5Fclose(static_cast<hid_t>(id_));
  id_ = -1;
}

bool H5File::exists(const std::string& path) const {
  if (path == "/") return true;
  INSEM_CHECK(!path.empty() && path[0] == '/', "HDF5 paths must be absolute: '" << path << "'");
  // H5Lexists requires every parent to exist, so walk the path.
  std::size_t pos = 0;
  while (pos != std::string::npos) {
    pos = path.find('/', pos + 1);
    const std::string prefix = path.substr(0, pos);
    if (H5Lexists(static_cast<hid_t>(id_), prefix.c_str(), H5P_DEFAULT) <= 0) return false;
  }
  return true;
}

void H5File::create_group(const std::string& path) {
  if (exists(path)) return;
  Handle lcpl = intermediate_groups();
  Handle group(H5Gcreate2(static_cast<hid_t>(id_), path.c_str(), lcpl.get(), H5P_DEFAULT,
                          H5P_DEFAULT),
               H5Gclose);
  INSEM_CHECK(group.get() >= 0, "cannot create group '" << path << "' in " << path_);
}

template <class T>
void H5File::write(const std::string& path, const T* data, const std::vector<std::size_t>& dims) {
  const hid_t file = static_cast<hid_t>(id_);
  if (exists(path)) H5Ldelete(file, path.c_str(), H5P_DEFAULT);
  std::vector<hsize_t> hdims(dims.begin(), dims.end());
  std::size_t count = 1;
  for (std::size_t d : dims) count *= d;
  Handle space(H5Screate_simple(static_cast<int>(hdims.size()), hdims.data(), nullptr), H5Sclose);
  INSEM_CHECK(space.get() >= 0, "cannot create dataspace for '" << path << "'");
  Handle lcpl = intermediate_groups();
  Handle dset(H5Dcreate2(file, path.c_str(), native_type<T>(), space.get(), lcpl.get(),
                         H5P_DEFAULT, H5P_DEFAULT),
              H5Dclose);
  INSEM_CHECK(dset.get() >= 0, "cannot create dataset '" << path << "' in " << path_);
  if (count > 0) {
    INSEM_CHECK(H5Dwrite(dset.get(), native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0,
                "cannot write dataset '" << path << "'");
  }
}

template <class T>
std::vector<T> H5File::read(const std::string& path, std::vector<std::size_t>* dims) const {
  INSEM_CHECK(exists(path), "dataset '" << path << "' not found in " << path_);
  Handle dset(H5Dopen2(static_cast<hid_t>(id_), path.c_str(), H5P_DEFAULT), H5Dclose);
  INSEM_CHECK(dset.get() >= 0, "cannot open dataset '" << path << "'");
  Handle space(H5Dget_space(dset.get()), H5Sclose);
  const int rank = H5Sget_simple_extent_ndims(space.get());
  INSEM_CHECK(rank >= 0, "cannot query dataspace of '" << path << "'");
  std::vector<hsize_t> hdims(rank);
  H5Sget_simple_extent_dims(space.get(), hdims.data(), nullptr);
  std::size_t count = 1;
  for (hsize_t d : hdims) count *= static_cast<std::size_t>(d);
  if (dims) dims->assign(hdims.begin(), hdims.end());
  std::vector<T> out(count);
  if (count > 0) {
    INSEM_CHECK(H5Dread(dset.get(), native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        out.data()) >= 0,
                "cannot read dataset '" << path << "'");
  }
  return out;
}

template void H5File::write<double>(const std::string&, const double*, const std::vector<std::size_t>&);
template void H5File::write<float>(const std::string&, const float*, const std::vector<std::size_t>&);
template void H5File::write<std::int32_t>(const std::string&, const std::int32_t*, const std::vector<std::size_t>&);
template void H5File::write<std::int64_t>(const std::string&, const std::int64_t*, const std::vector<std::size_t>&);
template std::vector<double> H5File::read<double>(const std::string&, std::vector<std::size_t>*) const;
template std::vector<float> H5File::read<float>(const std::string&, std::vector<std::size_t>*) const;
template std::vector<std::int32_t> H5File::read<std::int32_t>(const std::string&, std::vector<std::size_t>*) const;
template std::vector<std::int64_t> H5File::read<std::int64_t>(const std::string&, std::vector<std::size_t>*) const;

namespace {

// Writes a scalar attribute of `type`, replacing an existing one.
void write_attribute(hid_t file, const std::string& object, const std::string& name, hid_t type,
                     const void* value) {
  Handle obj(H5Oopen(file, object.c_str(), H5P_DEFAULT), H5Oclose);
  INSEM_CHECK(obj.get() >= 0, "cannot open object '" << object << "' for attribute '" << name << "'");
  if (H5Aexists(obj.get(), name.c_str()) > 0) H5Adelete(obj.get(), name.c_str());
  Handle space(H5Screate(H5S_SCALAR), H5Sclose);
  Handle attr(H5Acreate2(obj.get(), name.c_str(), type, space.get(), H5P_DEFAULT, H5P_DEFAULT),
              H5Aclose);
  INSEM_CHECK(attr.get() >= 0, "cannot create attribute '" << name << "' on '" << object << "'");
  INSEM_CHECK(H5Awrite(attr.get(), type, value) >= 0, "cannot write attribute '" << name << "'");
}

Handle open_attribute(hid_t file, const std::string& object, const std::string& name) {
  Handle obj(H5Oopen(file, object.c_str(), H5P_DEFAULT), H5Oclose);
  INSEM_CHECK(obj.get() >= 0, "cannot open object '" << object << "'");
  INSEM_CHECK(H5Aexists(obj.get(), name.c_str()) > 0,
              "attribute '" << name << "' not found on '" << object << "'");
  Handle attr(H5Aopen(obj.get(), name.c_str(), H5P_DEFAULT), H5Aclose);
  INSEM_CHECK(attr.get() >= 0, "cannot open attribute '" << name << "' on '" << object << "'");
  return attr;
}

}  // namespace

void H5File::set_attribute(const std::string& object, const std::string& name,
                           const std::string& value) {
  Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
  H5Tset_size(type.get(), value.empty() ? 1 : value.size());
  H5Tset_strpad(type.get(), H5T_STR_NULLPAD);
  const std::string padded = value.empty() ? std::string(1, '\0') : value;
  write_attribute(static_cast<hid_t>(id_), object, name, type.get(), padded.data());
}

void H5File::set_attribute(const std::string& object, const std::string& name, std::int64_t value) {
  write_attribute(static_cast<hid_t>(id_), object, name, H5T_NATIVE_INT64, &value);
}

void H5File::set_attribute(const std::string& object, const std::string& name, double value) {
  write_attribute(static_cast<hid_t>(id_), object, name, H5T_NATIVE_DOUBLE, &value);
}

bool H5File::has_attribute(const std::string& object, const std::string& name) const {
  if (!exists(object)) return false;
  return H5Aexists_by_name(static_cast<hid_t>(id_), object.c_str(), name.c_str(), H5P_DEFAULT) > 0;
}

std::string H5File::attribute_string(const std::string& object, const std::string& name) const {
  Handle attr = open_attribute(static_cast<hid_t>(id_), object, name);
  Handle type(H5Aget_type(attr.get()), H5Tclose);
  INSEM_CHECK(H5Tget_class(type.get()) == H5T_STRING,
              "attribute '" << name << "' on '" << object << "' is not a string");
  if (H5Tis_variable_str(type.get()) > 0) {
    // Variable-length strings, as written by h5py by default.
    Handle mem(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(mem.get(), H5T_VARIABLE);
    char* raw = nullptr;
    INSEM_CHECK(H5Aread(attr.get(), mem.get(), &raw) >= 0, "cannot read attribute '" << name << "'");
    std::string out = raw ? raw : "";
    H5free_memory(raw);
    return out;
  }
  const std::size_t size = H5Tget_size(type.get());
  std::string out(size, '\0');
  INSEM_CHECK(H5Aread(attr.get(), type.get(), out.data()) >= 0, "cannot read attribute '" << name << "'");
  out.resize(out.find('\0') == std::string::npos ? size : out.find('\0'));
  return out;
}

std::int64_t H5File::attribute_int(const std::string& object, const std::string& name) const {
  Handle attr = open_attribute(static_cast<hid_t>(id_), object, name);
  std::int64_t value = 0;
  INSEM_CHECK(H5Aread(attr.get(), H5T_NATIVE_INT64, &value) >= 0, "cannot read attribute '" << name << "'");
  return value;
}

double H5File::attribute_double(const std::string& object, const std::string& name) const {
  Handle attr = open_attribute(static_cast<hid_t>(id_), object, name);
  double value = 0.0;
  INSEM_CHECK(H5Aread(attr.get(), H5T_NATIVE_DOUBLE, &value) >= 0, "cannot read attribute '" << name << "'");
  return value;
}

}  // namespace insem::io
