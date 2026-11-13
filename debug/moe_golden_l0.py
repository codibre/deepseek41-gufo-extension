import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGML_QUANT_SIZES
d='/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/'
f2=d+'DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00002-of-00006.gguf'
r=GGUFReader(f2)
T={t.name:t for t in r.tensors}

def load_exp(name, e):
    t=T[name]
    K,N = int(t.shape[0]), int(t.shape[1])
    bs,tsz=GGML_QUANT_SIZES[int(t.tensor_type)]
    q=np.asarray(t.data[e]).reshape(N, K//bs, 1, tsz)
    return dequantize(q,t.tensor_type).reshape(N,-1)

def corr(a,b): return float(np.corrcoef(a.astype(np.float64),b.astype(np.float64))[0,1])

rlg=np.fromfile('/tmp/hb_all_rlg_00.bin',dtype=np.float32).reshape(-1,192)
sel=np.fromfile('/tmp/hb_all_rsel_00.bin',dtype=np.int32).reshape(-1,6)
rw =np.fromfile('/tmp/hb_all_rw_00.bin',dtype=np.float32).reshape(-1,6)
ro =np.fromfile('/tmp/hb_all_ro_00.bin',dtype=np.float32).reshape(-1,5120)
fn =np.fromfile('/tmp/hb_all_fn_00.bin',dtype=np.float32).reshape(-1,5120)
i=rlg.shape[0]-1
x=fn[i].astype(np.float32)
acc=np.zeros(5120,dtype=np.float64)
for k in range(6):
    e=int(sel[i][k])
    G=load_exp('blk.0.ffn_gate_exps.weight',e); U=load_exp('blk.0.ffn_up_exps.weight',e)
    gg=np.clip(G@x,-10,10); uu=np.clip(U@x,-10,10)
    mid=(gg/(1.0+np.exp(-gg)))*uu*np.float32(rw[i][k])
    D=load_exp('blk.0.ffn_down_exps.weight',e)
    acc+=(D@mid).astype(np.float64)
    del G,U,D,mid
gold=acc.astype(np.float32)
print('routed_out corr %.5f rms g %.4f rt %.4f maxerr %.4f'%(corr(gold,ro[i]),
      np.sqrt((gold.astype(np.float64)**2).mean()),
      np.sqrt((ro[i].astype(np.float64)**2).mean()),
      np.abs(gold.astype(np.float64)-ro[i].astype(np.float64)).max()))
