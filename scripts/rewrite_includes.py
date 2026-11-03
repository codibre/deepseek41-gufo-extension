#!/usr/bin/env python3
"""Rewrite gufo root-relative includes in vendored engine sources."""
import pathlib, re
ROOT = pathlib.Path("src/model")
RULES = [
    (r'#include "src/models/deepseek_v4_flash/', '#include "model/'),
    (r'#include "src/core/gguf_reader.hpp"', '#include "gufo/ext/gguf_reader.hpp"'),
    (r'#include "src/core/crypto/sha256.hpp"', '#include "gufo/ext/sha256.hpp"'),
    (r'#include "src/core/crypto/', '#include "gufo/ext/crypto.hpp"'),
    (r'#include "src/core/mapped_prefetch.hpp"', '#include "gufo/ext/mapped_prefetch.hpp"'),
    (r'#include "src/core/reasoning.hpp"', '#include "gufo/ext/reasoning.hpp"'),
    (r'#include "src/core/sampling.hpp"', '#include "gufo/ext/sampling.hpp"'),
    (r'#include "src/core/session_mode.hpp"', '#include "gufo/ext/session_mode.hpp"'),
    (r'#include "src/core/hip/', '#include "model/core_compat/hip/'),
    (r'#include "src/core/text_sampling_defaults.hpp"', '#include "model/core_compat/text_sampling_defaults.hpp"'),
    (r'#include "src/cli/serve/logging.hpp"', '#include "model/logging_shim.hpp"'),
    (r'#include "src/core/model_config.hpp"', '#include "model/core_compat/model_config.hpp"'),
    (r'#include "src/core/diagnostics/', '#include "model/core_compat/diagnostics/'),
    (r'#include "src/core/text_sampling_defaults.hpp"', '#include "model/core_compat/text_sampling_defaults.hpp"'),
]
n = 0
for p in ROOT.rglob("*"):
    if p.suffix not in (".cpp", ".hpp", ".h", ".hip") or "core_compat" in p.parts:
        continue
    s = t = p.read_text()
    for pat, rep in RULES:
        t = re.sub(pat, rep, t)
    if t != s:
        p.write_text(t)
        n += 1
print("rewrote", n, "files")
