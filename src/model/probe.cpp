#include "model/probe.hpp"

#include <filesystem>

#include "model/gguf/ds41_reader.hpp"

namespace gufo::models::deepseek41 {
namespace {

bool IsDeepSeek41(const std::string& path) {
  try {
    if (!std::filesystem::is_regular_file(path)) {
      return false;
    }
    std::string err;
    const auto reader =
        gg::Ds41Reader::OpenFile(path, &err, /*map_data=*/false);
    return reader != nullptr &&
           reader->GetMetadataString("general.architecture").value_or("") ==
               "deepseek41";
  } catch (...) {
    return false;
  }
}

}  // namespace

bool ProbeDeepSeek41(std::string_view model_root) {
  return IsDeepSeek41(std::string(model_root));
}

bool ClaimDeepSeek41File(const std::string& model_path) {
  return IsDeepSeek41(model_path);
}

}  // namespace gufo::models::deepseek41
