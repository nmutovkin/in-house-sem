#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>

#include "insem/core/error.hpp"
#include "insem/io/hdf5.hpp"

using insem::io::H5File;

TEST_CASE("HDF5 datasets and attributes round-trip", "[io]") {
  const auto path = (std::filesystem::temp_directory_path() / "insem_test_io.h5").string();
  {
    H5File f(path, H5File::Mode::Truncate);
    const std::vector<double> d{1.5, -2.0, 3.25, 4.0, 5.0, 6.0};
    f.write("/a/b/doubles", d.data(), {2, 3});
    f.write("/ints32", std::vector<std::int32_t>{1, -2, 3});
    f.write("/ints64", std::vector<std::int64_t>{std::int64_t{1} << 40});
    f.write("/floats", std::vector<float>{0.5f});
    f.write("/empty", std::vector<double>{});
    f.write("/ints32", std::vector<std::int32_t>{9, 8});  // replace
    f.set_attribute("/", "name", std::string("insem"));
    f.set_attribute("/", "empty", std::string());
    f.set_attribute("/a", "count", std::int64_t{42});
    f.set_attribute("/a/b/doubles", "scale", 0.125);
    f.set_attribute("/a", "count", std::int64_t{43});  // replace
  }
  H5File f(path, H5File::Mode::Read);
  std::vector<std::size_t> dims;
  CHECK(f.read<double>("/a/b/doubles", &dims) == std::vector<double>{1.5, -2.0, 3.25, 4.0, 5.0, 6.0});
  CHECK(dims == std::vector<std::size_t>{2, 3});
  CHECK(f.read<std::int32_t>("/ints32") == std::vector<std::int32_t>{9, 8});
  CHECK(f.read<std::int64_t>("/ints64") == std::vector<std::int64_t>{std::int64_t{1} << 40});
  CHECK(f.read<float>("/floats") == std::vector<float>{0.5f});
  CHECK(f.read<double>("/empty").empty());
  CHECK(f.attribute_string("/", "name") == "insem");
  CHECK(f.attribute_string("/", "empty").empty());
  CHECK(f.attribute_int("/a", "count") == 43);
  CHECK(f.attribute_double("/a/b/doubles", "scale") == 0.125);
  CHECK(f.exists("/a/b"));
  CHECK_FALSE(f.exists("/a/c/d"));
  CHECK_FALSE(f.has_attribute("/a", "missing"));
  CHECK_THROWS_AS(f.read<double>("/nope"), insem::Error);
  std::filesystem::remove(path);
}

TEST_CASE("Opening a missing file throws", "[io]") {
  CHECK_THROWS_AS(H5File("/nonexistent/dir/file.h5", H5File::Mode::Read), insem::Error);
}
