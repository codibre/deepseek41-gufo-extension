#include "model/probe.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>

#include "model/gguf/ds41_reader.hpp"

namespace gufo::models::deepseek41 {
namespace {

bool IsDeepSeek41(const std::string& path) {
  try {
    if (!std::filesystem::is_regular_file(path)) {
      std::fprintf(stderr, "[ds41-claim] not regular file: %s\n", path.c_str());
      return false;
    }
    std::string err;
    const auto reader =
        gg::Ds41Reader::OpenFile(path, &err, /*map_data=*/false);
    if (reader == nullptr) {
      std::fprintf(stderr, "[ds41-claim] OpenFile failed: %s\n", err.c_str());
      return false;
    }
    const auto arch = reader->GetMetadataString("general.architecture").value_or("");
    if (arch != "deepseek41") {
      std::fprintf(stderr, "[ds41-claim] arch='%s'\n", std::string(arch).c_str());
    }
    return arch == "deepseek41";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[ds41-claim] exception: %s\n", e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "[ds41-claim] unknown exception\n");
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
