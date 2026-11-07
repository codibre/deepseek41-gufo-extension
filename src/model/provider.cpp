#include <gufo/ext/generated/deepseek41_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include <chrono>
#include <memory>

#include "model/probe.hpp"

namespace gufo::models::deepseek_v4_flash {
#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateDeepSeekTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

int RunDeepSeekBenchmark(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start);
int RunDeepSeekPrompt(const ext::TextPromptOptions& opt,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point model_load_start);
int RunDeepSeekChat(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point model_load_start);
#endif
std::optional<gufo::sampling::TextSamplingDefaults>
DeepSeekSamplingDefaults(const core::GgufReader& reader);
}  // namespace gufo::models::deepseek_v4_flash

void deepseek41_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_deepseek41::
                           kProvider_deepseek41_gsq);
  registry.AddTextProvider({
      .id = "deepseek41-gsq",
      .probe = &gufo::models::deepseek41::ProbeDeepSeek41,
      .probe_gguf = &gufo::models::deepseek41::ProbeDeepSeek41Gguf,
#if defined(ENGINE_ENABLE_HIP)
      .create = &gufo::models::deepseek_v4_flash::CreateDeepSeekTextRunner,
      .bench_driver =
          &gufo::models::deepseek_v4_flash::RunDeepSeekBenchmark,
      .prompt_driver = &gufo::models::deepseek_v4_flash::RunDeepSeekPrompt,
      .chat_driver = &gufo::models::deepseek_v4_flash::RunDeepSeekChat,
#endif
      .sampling_defaults =
          &gufo::models::deepseek_v4_flash::DeepSeekSamplingDefaults,
      // gufo#966: this container fuses 3-D IQ2_XXS expert slabs core cannot
      // size. Opened with allow_unsized_tensors those tensors expose `data`
      // and their reserved `size_bytes`; this provider reads them itself
      // (Ds41Reader remains the engine's source of tensor bytes), so core
      // lets probe_gguf see such files. The claim_file first-refusal hook
      // this extension previously used was retired by the same rework.
      .accepts_unsized_tensors = true,
  });
}
