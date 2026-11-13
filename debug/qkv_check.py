# Consistency: golden q/kv (from rt in/cf streams) vs rt dumped aq/akv, L1.
import numpy as np, glob
from gguf import GGUFReader
import gguf.quants as qq

D = '/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/'
IDX = {}
for f in sorted(glob.glob(D + '*.gguf')):
    r = GGUFReader(f, 'r')
    for t in r.tensors:
        IDX[t.name] = (f, int(t.data_offset), int(t.tensor_type))
    del r

_QB = {8: (34, 32), 13: (176, 256)}
_EN = {8: 'Q8_0', 13: 'Q5_K'}

def dq(name, rows, cols):
    path, off, ty = IDX[name]
    if ty == 0:
        return np.fromfile(path, dtype=np.float32, count=rows * cols, offset=off).reshape(rows, cols)
    b, blk = _QB[ty]
    raw = np.fromfile(path, dtype=np.uint8, count=rows * ((cols + blk - 1) // blk) * b, offset=off)
    return np.asarray(qq.dequantize(raw.reshape(rows, (cols + blk - 1) // blk * b),
                                    getattr(qq.GGMLQuantizationType, _EN[ty])))

def rmsn(x):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-20)

TH = 10000.0
E = 448

def rope_apply(x, ang):
    # x (..., rot); ang (..., rot/2) broadcastable against x pairs
    a = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = a[..., 0], a[..., 1]
    c, s = np.cos(ang), np.sin(ang)
    o = np.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1)
    return o.reshape(x.shape)

n = 5
L = 1
b = f'blk.{L}.'
inA = np.fromfile('/tmp/hb_all_in_01.bin', dtype=np.float32).reshape(n, 4, 5120)
aq = np.fromfile('/tmp/hb_cap/hb_all_aq_01.bin', dtype=np.float32).reshape(n, 64, 512)
akv = np.fromfile('/tmp/hb_cap/hb_all_akv_01.bin', dtype=np.float32).reshape(n, 512)
qa = dq(b + 'attn_q_a.weight', 1280, 5120)
qn = dq(b + 'attn_q_a_norm.weight', 1, 1280)[0]
qb = dq(b + 'attn_q_b.weight', 32768, 1280)
kvw = dq(b + 'attn_kv.weight', 512, 5120)
kvn = dq(b + 'attn_kv_a_norm.weight', 1, 512)[0]
anw = dq(b + 'attn_norm.weight', 1, 5120)[0]

pos = np.arange(n, dtype=np.float64)
angbase = pos[:, None] / TH ** (np.arange(0, 64, 2)[None, :] / 64.0)   # (n,32)

def rope_q(q):   # q (n,64,512)
    r = q.copy()
    r[:, :, E:] = rope_apply(q[:, :, E:], angbase[:, None, :])
    return r

def rope_kv(kv):  # kv (n,512)
    r = kv.copy()
    r[:, E:] = rope_apply(kv[:, E:], angbase)
    return r

for tag, cff in [('cf_00', '/tmp/hb_all_cf_00.bin'), ('cf_01', '/tmp/hb_all_cf_01.bin')]:
    cfP = np.fromfile(cff, dtype=np.float32).reshape(n, 24)
    mix = np.einsum('th,thd->td', cfP[:, :4], inA)
    x = rmsn(mix) * anw
    KVr = rope_kv(rmsn(x @ kvw.T) * kvn)
    qr = rmsn(x @ qa.T) * qn
    qp = rope_q((qr @ qb.T).reshape(n, 64, 512))
    print(tag, 'q: corr %.5f  kv: corr %.5f' % (
        np.corrcoef(qp.ravel(), aq.ravel())[0, 1],
        np.corrcoef(KVr.ravel(), akv.ravel())[0, 1]))
