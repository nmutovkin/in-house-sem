#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace insem {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace insem

// Throws insem::Error with a streamed message when `cond` is false.
#define INSEM_CHECK(cond, msg)                                        \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::ostringstream insem_check_os_;                             \
      insem_check_os_ << msg << " [" << __FILE__ << ":" << __LINE__   \
                      << "]";                                         \
      throw ::insem::Error(insem_check_os_.str());                    \
    }                                                                 \
  } while (0)
