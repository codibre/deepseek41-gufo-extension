// Probe: does the PR966 core GgufReader open the DSV4.1 split container?
// mmap-only; no weight bytes committed. Safe next to the resident model.
#include <cstdio>
#include <string>
#include "src/core/gguf_reader.hpp"

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: p <shard1.gguf>\n"); return 2; }
  std::string err;
  auto r = gufo::core::GgufReader::OpenFile(argv[1], &err);
  if (!r) { std::printf("OPEN_FAILED: %s\n", err.c_str()); return 1; }
  std::printf("OPEN_OK tensors=%llu regions=%zu\n",
              (unsigned long long)r->GetTensorCount(),
              r->GetMappedRegions().size());
  auto meta = r->GetMetadataString("general.architecture");
  std::printf("arch=%s split.count=%u\n",
              meta ? std::string(*meta).c_str() : "?",
              r->GetMetadataUint32("split.count").value_or(1u));
  auto tensors = r->GetTensors();
  for (size_t i = 0; i < tensors.size() && i < 3; ++i)
    std::printf("  [%zu] %s n_dims=%zu type=%d\n", i,
                std::string(tensors[i].name).c_str(),
                tensors[i].dimensions.size(), (int)tensors[i].type);
  auto last = tensors.back();
  std::printf("  [last] %s type=%d\n", std::string(last.name).c_str(),
              (int)last.type);
  // Touch the FIRST byte of a mid-model tensor (one page fault, not a load).
  for (size_t i = 0; i < tensors.size(); ++i) {
    const std::string n = std::string(tensors[i].name);
    if (n.find("blk.20.") != std::string::npos) {
      if (const void* p = tensors[i].data) {
        volatile unsigned char b = *static_cast<const unsigned char*>(p);
        std::printf("  mid-model tensor %s -> data OK (first byte %02x)\n",
                    n.c_str(), b);
      } else std::printf("  mid-model tensor %s -> NO DATA\n", n.c_str());
      break;
    }
  }
  return 0;
}
