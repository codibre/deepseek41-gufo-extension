#!/usr/bin/env python3
"""Golden-vs-runtime engram check for CAP prefill (L1=mod0, L14=mod1).
Uses gdump2 (bit-exact ggml dequant) for engram_kv Q5_K weights and
reproduces ds4_engram_kv_kernel + ds4_engram_gate_kernel in numpy."""
import sys, os, subprocess, tempfile
import numpy as np
sys.path.insert(0, '/home/tosol/projects/personal/llama.cpp/gguf-py')
from gguf import GGUFReader

BASE = '/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-{:05d}-of-00006.gguf'
GD = '/home/tosol/projects/personal/deepseek41-gufo-extension/debug/gdump2'
EMB, HC, IN = 5120, 4, 6144
OUT = (HC + 1) * EMB
EPS = 1e-20

_readers = {}
def find_tensor(name):
    for i in (2, 3, 4, 5, 6):
        if i not in _readers:
            _readers[i] = GGUFReader(BASE.format(i))
        for t in _readers[i].tensors:
            if t.name == name:
                return i, t
    raise KeyError(name)

def dq(name, typ, rows, cols):
    i, t = find_tensor(name)
    assert t.tensor_type.name.lower().replace('_', '') == typ.replace('_', ''), (t.tensor_type.name, typ)
    nb = t.n_bytes
    with open(BASE.format(i), 'rb') as f:
        f.seek(t.data_offset)
        buf = f.read(nb)
    tmp = tempfile.mktemp(suffix='.bin')
    open(tmp, 'wb').write(buf)
    r = subprocess.run([GD, tmp, typ, str(rows), str(cols)], capture_output=True)
    os.unlink(tmp)
    if r.returncode:
        raise RuntimeError(f'gdump2 fail {name}: {r.stderr[:200]}')
    a = np.frombuffer(r.stdout, dtype=np.float32)
    assert a.size == rows * cols, (name, a.size, rows * cols)
    return a.reshape(rows, cols).copy()

def ppath(kind, il):
    for s in (f'{il:02d}', f'{il}'):
        f = f'/tmp/eg_{kind}_L{s}.bin'
        if os.path.exists(f):
            return f
    raise SystemExit(f'missing {kind} L{il}')

def check(il):
    rows = np.fromfile(ppath('rows', il), dtype=np.float32)
    kv   = np.fromfile(ppath('kv', il),   dtype=np.float32)
    pre  = np.fromfile(ppath('pre', il),  dtype=np.float32)
    post = np.fromfile(ppath('post', il), dtype=np.float32)
    gw   = np.fromfile(ppath('gatew', il),dtype=np.float32)
    nt = len(rows) // IN
    print(f'--- L{il}: nt={nt} bytes rows/kv/pre/post/gw = {len(rows)}/{len(kv)}/{len(pre)}/{len(post)}/{len(gw)}')
    rows = rows.reshape(nt, IN)
    kv = kv.reshape(nt, OUT)
    pre = pre.reshape(nt, HC, EMB)
    post = post.reshape(nt, HC, EMB)
    gw = gw.reshape(-1)
    # kernel indexes egate[h*n_embd + d] -> row-major [4][5120]
    gwl = gw.reshape(HC, EMB)
    # rows sanity vs rms
    print(f'rows rms {np.sqrt((rows**2).mean()):.4f}  kv rms {np.sqrt((kv**2).mean()):.4f}  pre rms {np.sqrt((pre**2).mean()):.4f}')
    # golden kv
    W = dq(f'blk.{il}.engram_kv.weight', 'q5_k', OUT, IN)
    kv_g = rows @ W.T
    err = np.abs(kv_g - kv)
    rel = err.max() / (np.abs(kv_g).max() + 1e-30)
    print(f'kv: max abs err {err.max():.5f} (golden max {np.abs(kv_g).max():.3f}) rel {rel:.4f} corr {np.corrcoef(kv_g.ravel(), kv.ravel())[0,1]:.5f}')
    # golden gate
    val = kv[:, HC * EMB:].reshape(nt, EMB)
    inv = 1.0 / np.sqrt(EMB)
    sg = np.zeros((nt, HC), dtype=np.float32)
    for t in range(nt):
        for h in range(HC):
            xh = pre[t, h]
            kh = kv[t, h * EMB:(h + 1) * EMB]
            rs1 = 1.0 / np.sqrt((xh * xh).mean() + EPS)
            rs2 = 1.0 / np.sqrt((kh * kh).mean() + EPS)
            dot = float((xh * kh * gwl[h]).sum()) * rs1 * rs2 * inv
            mag = np.sqrt(max(abs(dot), 1e-6))
            sg[t, h] = 1.0 / (1.0 + np.exp(-np.copysign(mag, dot)))
    post_g = pre.copy()
    for t in range(nt):
        for h in range(HC):
            post_g[t, h] = pre[t, h] + val[t] * sg[t, h]
    perr = np.abs(post_g - post)
    print(f'gate: sg[0]={np.round(sg[0],4)}  post max abs err {perr.max():.6f}  post rms {np.sqrt((post**2).mean()):.4f}')
    # runtime-implied gate: (post-pre)/val, median over dims with |val|>0.5
    delta3 = (post - pre)
    mask = np.abs(val)[:, None, :] > 0.5
    print('implied gate vs golden sg:')
    for t in range(nt):
        row = []
        for h in range(HC):
            m = mask[t, 0]
            row.append(round(float(np.median(delta3[t, h][m])), 4) if m.any() else float('nan'))
        print(f'  t{t} implied={row}  golden={list(np.round(sg[t], 4))}')
    # applied delta magnitude vs stream
    delta = post - pre
    print(f'delta rms {np.sqrt((delta**2).mean()):.4f}  (pre rms {np.sqrt((pre**2).mean()):.4f})')

for il in (1, 14):
    check(il)
