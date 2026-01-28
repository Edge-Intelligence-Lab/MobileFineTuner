#include "logger.h"

namespace ops {

// Static member definition
std::unique_ptr<Logger> LogManager::instance_ = nullptr;

} // namespace ops
