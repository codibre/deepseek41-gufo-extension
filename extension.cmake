# DeepSeek-V4.1-Flash (deepseek41 + Engram) model extension for gufo.
#
# Engine vendored from gufo src/models/deepseek_v4_flash (in-tree at the
# pinned base) + a split-aware GGUF reader and claim_file probe. The core
# GGUF loader rejects this container (metadata-only shard 1, fused IQ2_XXS
# expert slabs), so the extension claims it and reads it end-to-end.
# Out-of-tree rules: link gufo_ext_runtime only; never gufo_core.

set(DS41_DIR "${GUFO_EXTENSION_BUILD_ROOT}/src/model")
set(DS41_RUNTIME_DIR "${DS41_DIR}/runtime")
set(DS41_ROCM_DIR "${DS41_DIR}/kernels/rocm")
set(DS41_MMQ_DIR "${DS41_ROCM_DIR}/mmq")

# Vendored llama.cpp quantized-matmul tier (own C++17 dialect + GGML stubs;
# see the in-tree VENDOR.md carried alongside).
set(DS41_MMQ_SOURCES
  ${DS41_MMQ_DIR}/ds4_ggml_stubs.hip.cpp
  ${DS41_MMQ_DIR}/ds4_mmq.hip.cpp
  ${DS41_MMQ_DIR}/ds4_mmq_d2r.hip.cpp
  ${DS41_MMQ_DIR}/mmid.hip.cpp
  ${DS41_MMQ_DIR}/mmvq.hip.cpp
  ${DS41_MMQ_DIR}/quantize.hip.cpp)

if(TARGET gufo_deepseek41 AND ENGINE_ENABLE_HIP)
  add_library(gufo_deepseek41_mmq STATIC ${DS41_MMQ_SOURCES})
  set_source_files_properties(${DS41_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_deepseek41_mmq PROPERTIES
    HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_deepseek41_mmq PRIVATE
    GGML_USE_HIP GGML_HIP_NO_VMM DS4_ROCM_WMMA_MMQ_Y=64)
  target_compile_options(gufo_deepseek41_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-value>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
  target_include_directories(gufo_deepseek41_mmq SYSTEM PUBLIC ${DS41_MMQ_DIR})
  target_link_libraries(gufo_deepseek41_mmq PUBLIC
    gufo_ext_runtime hip::host roc::hipblas)
endif()

# Host tier: registration, claim/probe, split-aware reader, chat template.
gufo_extension_target(
  NAME gufo_deepseek41
  SOURCES
    src/model/extension.cpp
    src/model/probe.cpp
    src/model/gguf/ds41_reader.cpp
    src/model/chat_template.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_deepseek41)
  target_include_directories(gufo_deepseek41 PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}/src>"
  )
endif()

# Device tier: the vendored deepseek_v4_flash engine + HIP runtime plumbing.
gufo_extension_target(
  NAME gufo_deepseek41_hip
  SOURCES
    src/model/engine.cpp
    src/model/provider.cpp
    src/model/runtime/model_data.cpp
    src/model/runtime/rocm_graph.cpp
    src/model/runtime/sampling.cpp
    src/model/runtime/session.cpp
    src/model/runtime/tokenizer.cpp
    src/model/core_compat/hip/bandwidth.cpp
    src/model/core_compat/diagnostics/fingerprint.cpp
    src/model/core_compat/diagnostics/system_inventory.cpp
    src/model/core_compat/hip/weight_upload.cpp
    src/model/core_compat/hip/hipblaslt_plan_database.cpp
  LINK_LIBRARIES
    gufo_deepseek41
    gufo_deepseek41_mmq
    gufo_ext_runtime
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_deepseek41_hip AND ENGINE_ENABLE_HIP)
  gufo_extension_hip_sources(gufo_deepseek41_hip
    SOURCES
      src/model/kernels/rocm/backend.hip.cpp
  )
  target_compile_definitions(gufo_deepseek41_hip PRIVATE
    _GNU_SOURCE DS4_ROCM_BUILD=1 ENGINE_ENABLE_HIP=1 GGML_HIP_NO_VMM)
  target_compile_options(gufo_deepseek41_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
  target_include_directories(gufo_deepseek41_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}/src>"
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  target_include_directories(gufo_deepseek41_hip SYSTEM PUBLIC
    ${DS41_RUNTIME_DIR}
    ${DS41_ROCM_DIR}
    ${DS41_ROCM_DIR}/detail
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
endif()
