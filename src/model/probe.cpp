#include "model/probe.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>

#include "gufo/ext/sampling.hpp"
#include "gufo/ext/text.hpp"
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

bool ProbeDeepSeek41Gguf(const gufo::core::GgufReader& reader) {
  // Reached only for files core opened with allow_unsized_tensors (the fused
  // IQ2_XXS expert slabs of this container). Architecture is the gate; the
  // engine reads tensor bytes via Ds41Reader, never the core tensor table.
  return reader.GetMetadataString("general.architecture").value_or("") ==
         "deepseek41";
}

}  // namespace gufo::models::deepseek41

namespace gufo::models::deepseek_v4_flash {

std::optional<gufo::sampling::TextSamplingDefaults> DeepSeekSamplingDefaults() {
  return gufo::sampling::PresetDefaults(
      gufo::sampling::TextModelPreset::kDeepSeekV4Flash);
}

std::optional<gufo::sampling::TextSamplingDefaults> DeepSeekSamplingDefaults(
    const gufo::core::GgufReader&) {
  return DeepSeekSamplingDefaults();
}

}  // namespace gufo::models::deepseek_v4_flash
