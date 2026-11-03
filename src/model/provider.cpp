#include <gufo/ext/generated/deepseek41_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "model/probe.hpp"

void deepseek41_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_deepseek41::
                           kProvider_deepseek41_gsq);
  registry.AddTextProvider({
      .id = "deepseek41-gsq",
      .probe = &gufo::models::deepseek41::ProbeDeepSeek41,
      // First refusal: gufo core has no deepseek41 architecture and its
      // strict GGUF loader rejects this container's split layout and fused
      // IQ2_XXS expert slabs. Claiming makes this provider own the file end
      // to end (the core hands us an empty reader by contract; Ds41Reader
      // is the single source of tensor bytes). create/drivers land with the
      // vendored engine (docs/PORT-NOTES.md phase B+).
      .claim_file = &gufo::models::deepseek41::ClaimDeepSeek41File,
  });
}
