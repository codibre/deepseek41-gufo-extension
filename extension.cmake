# DeepSeek V4.1 Flash ("deepseek41", Engram) out-of-tree model extension.
#
# The core GGUF loader rejects this container: multi-shard files behind a
# metadata-only shard 1 and IQ2_XXS-fused 3-D expert slabs. The extension
# claims deepseek41 artifacts via TextProvider::claim_file and reads them
# with its own reader. Out-of-tree rules: link gufo_ext_runtime only.

gufo_extension_target(
  NAME gufo_deepseek41
  SOURCES
    src/model/gguf/ds41_reader.cpp
    src/model/probe.cpp
    src/model/extension.cpp
    src/model/provider.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_deepseek41)
  target_include_directories(gufo_deepseek41 PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}/src>"
  )
endif()
