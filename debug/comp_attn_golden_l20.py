# Golden: L02 mixed-window attention (raw + compressed rows) vs runtime ho_02.
# Kernel: attention_static_mixed_heads8_online_kernel(t,n_comp=88,win=128,ratio=2)
# visibility: raw [max(0,t+1-128), t]; comp [0, min((t+1)/2, n_comp, wstart/2))
import numpy as np
from gguf import GGUFReader

NT, NH, HD, WIN, RATIO, NCOMP = 176, 64, 512, 128, 1, 176
q   = np.fromfile('/tmp/hb_all_qr.bin', dtype=np.float32).reshape(NT, NH, HD)
kv  = np.fromfile('/tmp/hb_all_kv.bin', dtype=np.float32).reshape(NT, HD)
cc  = np.fromfile('/tmp/hb_all_cc_20.bin', dtype=np.float32).reshape(NCOMP, HD)
ho  = np.fromfile('/tmp/hb_all_ho_20.bin', dtype=np.float32).reshape(NT, NH, HD)

r = GGUFReader('/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00003-of-00006.gguf')
T = {t.name: t for t in r.tensors}
tt = T['blk.20.attn_sinks.weight']
print('sink dtype:', tt.data.dtype if hasattr(tt.data,'dtype') else type(tt.data))
snk = np.asarray(tt.data, dtype=np.float32).copy()
assert snk.size == NH, snk.shape
scale = 1.0 / np.sqrt(np.float32(HD))

def vis_comp(t):
    v = (t + 1) // RATIO
    v = min(v, NCOMP)
    wstart = t + 1 - WIN if t + 1 > WIN else 0
    return min(v, wstart // RATIO)

worst = []
for t in range(NT):
    rs = max(0, t + 1 - WIN)
    K = kv[rs:t+1]                       # (rc, HD) raw rows
    vc = vis_comp(t)
    C = cc[:vc] if vc else np.zeros((0, HD), np.float32)
    Q = q[t]                             # (NH, HD)
    S = np.concatenate([ (Q @ K.T) * scale,
                         (Q @ C.T) * scale if vc else np.zeros((NH,0), np.float32) ], axis=1)
    m  = np.maximum(S.max(axis=1), snk)  # (NH,)
    E  = np.exp(S - m[:, None])
    den = E.sum(axis=1) + np.exp(snk - m)
    V  = np.concatenate([K, C], axis=0)  # shared K=V (MLA)
    o  = (E @ V) / den[:, None]
    rt = ho[t]
    d  = np.abs(rt - o).max()
    c  = np.corrcoef(rt.ravel(), o.ravel())[0,1]
    rms_g = np.sqrt((o**2).mean()); rms_r = np.sqrt((rt**2).mean())
    worst.append((t, vc, c, d, rms_g, rms_r))

for row in worst[-3:] + [worst[0]]:
    t, vc, c, d, g, rr = row
    print(f't={t:3d} comp={vc:2d} corr {c:.5f} maxerr {d:.5f} rms g {g:.4f} rt {rr:.4f}')
bad = [w for w in worst if w[2] < 0.9999]
first_bad = bad[0][0] if bad else None
print(f'rows={len(worst)} rows corr<0.9999: {len(bad)} first_bad_t={first_bad}')
