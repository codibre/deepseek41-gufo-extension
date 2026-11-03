// Standalone Ds41Reader harness (no gufo runtime needed at run time).
// Build (headers from a configured gufo tree; see scripts/build.sh):
//   BG=build-gufo
//   g++ -std=c++20 -O1 -I"$BG/include" -I"$BG" -Isrc \
//       src/model/gguf/ds41_reader.cpp scripts/inspect.cpp -o /tmp/ds41_inspect
// Run:  /tmp/ds41_inspect <shard1.gguf> [--map]
#include <cstdio>
#include <string>
#include <vector>

#include "model/gguf/ds41_reader.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <shard1.gguf> [--map]\n", argv[0]);
    return 2;
  }
  const bool map_data = argc > 2 && std::string(argv[2]) == "--map";
  std::string err;
  auto reader = gufo::models::deepseek41::gg::Ds41Reader::OpenFile(
      argv[1], &err, map_data);
  if (reader == nullptr) {
    std::fprintf(stderr, "open failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("arch=%s name=%s\n",
              std::string(reader->GetMetadataString("general.architecture")
                              .value_or("?")).c_str(),
              std::string(reader->GetMetadataString("general.name")
                              .value_or("?")).c_str());
  std::printf("n_tensors=%zu total_bytes=%.2f GB mapped_regions=%zu\n",
              reader->Tensors().size(), reader->GetSize() / 1e9,
              reader->GetMappedRegions().size());
  for (const char* key :
       {"split.count", "split.tensors.count", "deepseek41.block_count",
        "deepseek41.embedding_length", "deepseek41.attention.head_count",
        "deepseek41.attention.head_count_kv",
        "deepseek41.attention.q_lora_rank",
        "deepseek41.attention.output_lora_rank",
        "deepseek41.attention.output_group_count",
        "deepseek41.attention.indexer.head_count",
        "deepseek41.attention.indexer.top_k",
        "deepseek41.attention.indexer.candidate_source_layer",
        "deepseek41.attention.indexer.candidate_top_k_blocks",
        "deepseek41.attention.indexer.candidate_block_size",
        "deepseek41.expert_count", "deepseek41.expert_used_count",
        "deepseek41.expert_feed_forward_length",
        "deepseek41.expert_shared_count",
        "deepseek41.hyper_connection.count",
        "deepseek41.ple.ngram_size", "deepseek41.ple.heads_per_ngram",
        "deepseek41.embedding_length_per_layer_input",
        "deepseek41.engram.pad_token_id", "deepseek41.context_length"}) {
    if (auto v = reader->GetMetadataUint32(key)) {
      std::printf("%s=%u\n", key, *v);
    } else if (auto w = reader->GetMetadataUint64(key)) {
      std::printf("%s=%llu\n", key, (unsigned long long)*w);
    } else if (auto s = reader->GetMetadataString(key)) {
      std::printf("%s=%s\n", key, std::string(*s).c_str());
    }
  }
  for (const char* name :
       {"token_embd.weight", "output.weight", "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_down_exps.weight", "blk.1.engram_kv.weight",
        "blk.1.engram_gate.weight", "blk.14.engram_kv.weight",
        "per_layer_token_embd.weight"}) {
    const auto* t = reader->FindTensor(name);
    if (t == nullptr) {
      std::printf("MISSING tensor %s\n", name);
      continue;
    }
    std::printf("%s type=%d dims=[", name, (int)t->type);
    for (auto d : t->dimensions) std::printf("%llu,", (unsigned long long)d);
    std::printf("] %.2f MB data=%s\n", t->size_bytes / 1e6,
                t->data != nullptr ? "yes" : "no");
  }
  for (const char* key : {"deepseek41.attention.kv_source_layers",
                          "deepseek41.attention.indexer.source_layers",
                          "deepseek41.ple.layers"}) {
    auto v = reader->GetMetadataUint32Array(key);
    std::printf("%s(n=%zu):", key, v.size());
    for (size_t i = 0; i < v.size() && i < 12; ++i) std::printf(" %u", v[i]);
    std::printf("\n");
  }
  auto tm = reader->GetMetadataUint32Array("deepseek41.engram.token_map");
  std::printf("engram.token_map entries=%zu head=[%u,%u,%u]\n", tm.size(),
              tm.size() > 0 ? tm[0] : 0, tm.size() > 1 ? tm[1] : 0,
              tm.size() > 2 ? tm[2] : 0);
  return 0;
}
