# V1 over rt aq/akv, brute rope conventions, vs rt ho and ay (L1, CAP P=4).
import numpy as np
exec(open('/home/tosol/projects/personal/deepseek41-gufo-extension/debug/qkv_check.py').read().split('for tag, cff')[0])

n = 5
P = 4
aq = np.fromfile('/tmp/hb_cap/hb_all_aq_01.bin', dtype=np.float32).reshape(n, 64, 512)
akv = np.fromfile('/tmp/hb_cap/hb_all_akv_01.bin', dtype=np.float32).reshape(n, 512)
ho = np.fromfile('/tmp/hb_cap/hb_all_ho_01.bin', dtype=np.float32).reshape(n, 64, 512)[P].ravel()
ay = np.fromfile('/tmp/hb_cap/hb_ay_L01.bin', dtype=np.float32)
sink = np.load("/tmp/sink_l1.npy").astype(np.float64)
sc = 1 / np.sqrt(512.0)

def unrope_q(q):
    q = q.copy()
    r = q[:, :, 448:].reshape(n, 64, 32, 2)
    c = np.cos(angbase)[:, None, :]
    sn = np.sin(angbase)[:, None, :]
    x1, x2 = r[..., 0], r[..., 1]
    q[:, :, 448:] = np.stack([x1 * c + x2 * sn, -x1 * sn + x2 * c], -1).reshape(n, 64, 64)
    return q

def unrope_kv(k):
    k = k.copy()
    r = k[:, 448:].reshape(n, 32, 2)
    c = np.cos(angbase)
    sn = np.sin(angbase)
    x1, x2 = r[..., 0], r[..., 1]
    k[:, 448:] = np.stack([x1 * c + x2 * sn, -x1 * sn + x2 * c], -1).reshape(n, 64)
    return k

def V1(q, k):
    s = np.einsum('thd,kd->htk', q.astype(np.float64), k.astype(np.float64)) * sc  # (64,n,n)
    out = np.zeros((64, 512))
    for h in range(64):
        sm = max(s[h, P, :P + 1].max(), sink[h])
        e = np.exp(s[h, P, :P + 1] - sm)
        den = e.sum() + np.exp(sink[h] - sm)
        out[h] = (e @ k[:P + 1]) / den
    return out.ravel()

for qn_, q in [('as-is', aq), ('unrope', unrope_q(aq))]:
    for kn_, k in [('as-is', akv), ('unrope', unrope_kv(akv))]:
        y = V1(q, k)
        print(qn_, kn_, 'corr ho %.5f | ''| rms y %.4f ho %.4f ay %.4f' % (
            np.corrcoef(y, ho)[0, 1], 
            np.sqrt((y * y).mean()), np.sqrt((ho * ho).mean()), np.sqrt((ay * ay).mean())))
