#ifndef GUFO_MODELS_DEEPSEEK41_PROBE_HPP_
#define GUFO_MODELS_DEEPSEEK41_PROBE_HPP_

#include <string>
#include <string_view>

namespace gufo::models::deepseek41 {

/// True when `model_root` (a GGUF path) is a DeepSeek V4.1 container
/// (general.architecture == "deepseek41"). Opens headers only.
[[nodiscard]] bool ProbeDeepSeek41(std::string_view model_root);

/// First-refusal claim: gufo core has no deepseek41 provider (unknown
/// architecture), and its strict GGUF loader additionally rejects the split
/// container's metadata-only shard 1 and the IQ2_XXS-fused 3-D expert slabs.
/// Every deepseek41 artifact therefore belongs to this extension.
[[nodiscard]] bool ClaimDeepSeek41File(const std::string& model_path);

}  // namespace gufo::models::deepseek41

#endif  // GUFO_MODELS_DEEPSEEK41_PROBE_HPP_
