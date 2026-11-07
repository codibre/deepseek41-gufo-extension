#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "gufo/ext/sampling.hpp"
#include "gufo/ext/text.hpp"

namespace gufo::models::deepseek41 {

// Phase A (docs/PORT-NOTES.md): probe reports the model family. gufo#966
// rework: the core opens this container with `allow_unsized_tensors` (its
// fused IQ2_XXS expert slabs are outside the core quant table) and lets
// providers that opt in via `accepts_unsized_tensors` probe the reader;
// Ds41Reader remains the single source of tensor bytes for the engine.
// Signatures match gufo::ext::TextProbe / gufo::ext::TextProbeGguf.
[[nodiscard]] bool ProbeDeepSeek41(std::string_view model_root);
[[nodiscard]] bool ProbeDeepSeek41Gguf(const gufo::core::GgufReader& reader);

}  // namespace gufo::models::deepseek41

namespace gufo::models::deepseek_v4_flash {

// Sampling defaults shared by the runner descriptor and the provider probe
// (mirrors the in-tree deepseek_v4_flash probe at the pinned head).
[[nodiscard]] std::optional<gufo::sampling::TextSamplingDefaults>
DeepSeekSamplingDefaults();
[[nodiscard]] std::optional<gufo::sampling::TextSamplingDefaults>
DeepSeekSamplingDefaults(const gufo::core::GgufReader& reader);

}  // namespace gufo::models::deepseek_v4_flash
