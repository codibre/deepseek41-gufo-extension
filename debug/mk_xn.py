# Build golden attention input rows for layer L:
#   x[t] = rmsn( sum_h cf_{L}[t][h] * in_L[t][h] ) * attn_norm_weight
# (the n×24 comb dumped at layer L is the one that mixes in_L; proven at L0
#  by g0_chain_v3 corr 0.99992 using ca@L0 x in_00)
import numpy as np, sys, glob, os
L=int(sys.argv[1]); n=int(sys.argv[2])
D=5120
inA=np.fromfile(f'/tmp/hb_all_in_{L:02d}.bin',dtype=np.float32).reshape(n,4,D)
cfP=np.fromfile(f'/tmp/hb_all_cf_{L:02d}.bin',dtype=np.float32).reshape(n,24)
mix=np.einsum('th,thd->td',cfP[:,:4],inA)
xn=mix/np.sqrt((mix*mix).mean(axis=1,keepdims=True)+1e-20)
# attn_norm weight
from gguf import GGUFReader
IDX={}
for f in glob.glob('/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/*.gguf'):
    r=GGUFReader(f,'r')
    for t in r.tensors:
        if t.name==f'blk.{L}.attn_norm.weight':
            w=np.frombuffer(bytes(t.data),dtype=np.float32).copy(); print('w rms range',float(w.min()),float(w.max()))
            break
    else: continue
    break
out=xn*w
out.astype(np.float32).tofile(f'/tmp/hg_xn_L{L:02d}.bin')
print('wrote', f'/tmp/hg_xn_L{L:02d}.bin', out.shape, 'rms', float((out**2).mean()**.5))
