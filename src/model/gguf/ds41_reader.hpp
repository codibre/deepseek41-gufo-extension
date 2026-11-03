#ifndef GUFO_MODELS_DEEPSEEK41_DS41_READER_HPP_
#define GUFO_MODELS_DEEPSEEK41_DS41_READER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

namespace gufo::models::deepseek41::gg {

// Container-level GGUF records shared with core (plain data; core performs no
// quant validation on them). Aliased so the reader stays self-contained.
using GgufTensor = gufo::core::GgufTensorInfo;
using GgufRegion = gufo::core::GgufMappedRegion;
using GgufMeta = gufo::core::GgufMetadataValue;
using GgufValueType = gufo::core::GgufValueType;
using GgmlType = gufo::core::GgmlType;

/// Standalone GGUF container reader.
///
/// Parses the header, metadata and tensor table without validating tensor
/// storage against any quant type table: each tensor's byte size is derived
/// from the container's own offset layout (tensor bytes run to the next
/// tensor's offset; the last tensor of a shard runs to the end of that
/// shard's data region). That is exactly what a claiming extension needs to
/// load files the core GGUF loader rejects for unknown quant types or fused
/// expert slab layouts — the extension's kernels interpret the bytes.
///
/// Returns the same public `GgufTensor`, `GgufMeta` and
/// `GgufRegion` views as the core reader, so model code is agnostic
/// to which reader produced them.
class Ds41Reader {
 public:
  /// Opens `path` (shard `00001` of a split container, or a single file).
  /// With `map_data` true every shard is mmap'd whole and tensor `data`
  /// pointers resolve; with false only headers are read (claim/probe use).
  [[nodiscard]] static std::unique_ptr<Ds41Reader> OpenFile(
      const std::string& path, std::string* error_msg, bool map_data = true);

  ~Ds41Reader();
  Ds41Reader(const Ds41Reader&) = delete;
  Ds41Reader& operator=(const Ds41Reader&) = delete;

  [[nodiscard]] const GgufTensor* FindTensor(
      std::string_view name) const noexcept;
  [[nodiscard]] std::span<const GgufTensor> Tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] std::span<const GgufRegion> GetMappedRegions()
      const noexcept {
    return regions_;
  }
  [[nodiscard]] const GgufMeta* FindMetadata(
      std::string_view key) const noexcept;
  [[nodiscard]] std::optional<std::string_view> GetMetadataString(
      std::string_view key) const;
  [[nodiscard]] std::optional<std::uint32_t> GetMetadataUint32(
      std::string_view key) const;
  /// F32 metadata (a stored F64 narrows to F32).
  [[nodiscard]] std::optional<float> GetMetadataFloat32(
      std::string_view key) const;
  [[nodiscard]] std::optional<std::uint64_t> GetMetadataUint64(
      std::string_view key) const;
  /// F32 metadata (a stored F64 narrows to F32).
  [[nodiscard]] std::vector<std::string_view> GetMetadataStringArray(
      std::string_view key) const;
  /// Typed array accessors for V4.1 config arrays (ple_layer_ids,
  /// kv_source_per_layer, topk_source_per_layer, candidate_per_layer,
  /// swiglu_limits, ...). Parsed arrays widen to uint64/int64/double;
  /// these narrow to the engine's types.
  [[nodiscard]] std::vector<std::uint32_t> GetMetadataUint32Array(
      std::string_view key) const;
  [[nodiscard]] std::vector<float> GetMetadataFloat32Array(
      std::string_view key) const;

  /// Total bytes across all mapped (or known) shard regions.
  [[nodiscard]] std::uint64_t GetSize() const noexcept;

 private:
  Ds41Reader() = default;

  struct Shard {
    std::filesystem::path path;
    void* mapped{nullptr};
    std::size_t mapped_size{0};
    gufo::platform::PlatformFile fd{gufo::platform::kInvalidFile};
    std::vector<char> blob;  // owned header bytes when not mapped
    std::uint64_t data_start{0};
  };

  bool ParseShard(Shard& shard, std::string* error_msg);

  std::vector<Shard> shards_;
  std::vector<GgufRegion> regions_;
  std::vector<GgufTensor> tensors_;
  std::map<std::string, GgufMeta, std::less<>> metadata_;
};

/// Model-wide alias so engine code is reader-agnostic.
using ModelGguf = Ds41Reader;


}  // namespace gufo::models::deepseek41::gg

namespace gufo::models::deepseek41 {
using gg::ModelGguf;
}  // namespace gufo::models::deepseek41

#endif  // GUFO_MODELS_DEEPSEEK41_DS41_READER_HPP_
