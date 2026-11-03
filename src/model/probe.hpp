#pragma once

#include <string>
#include <string_view>

#include "gufo/ext/text.hpp"

namespace gufo::models::deepseek41 {

// Phase A (docs/PORT-NOTES.md): probe reports the model family; claim_file
// makes this provider own deepseek41 GGUF containers end to end, before the
// core loader rejects their split layout / fused IQ2_XXS expert slabs.
// Signatures match gufo::ext::TextProbe / gufo::ext::TextFileClaim.
[[nodiscard]] bool ProbeDeepSeek41(std::string_view model_root);
bool ClaimDeepSeek41File(const std::string& model_path);

}  // namespace gufo::models::deepseek41

