# reference/

`deepseek-inference/` is a verbatim copy of the **official** DeepSeek
reference inference code from
<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference>
(MIT, © DeepSeek-AI), fetched 2026-11-03 (model.py 61669 B, engram.py 8138 B,
kernel.py 23790 B, generate.py 8722 B, config.json 1982 B, README.md 2029 B).

Kept for porting semantics (Engram, CSA2/indexer, Hyper-Connections, chat
format) — it is NOT part of the build. The gufo engine implements this in
C++/HIP; llama.cpp `src/models/deepseek41.cpp` is the closest C++ port.
