# Golden attention chain for ANY layer L and ANY prompt (generalizes v3):
# weights located via GGUFReader over all shards; x (stream collapsed+normed)
# provided from a file (row P of attention-input post-norm), tokens from hb.
import numpy as np, subprocess, os, tempfile, sys
D_DIR='/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/'
GD=os.path.expanduser('~/.hermes/cache/scratch/gdump2')
from gguf import GGUFReader
import glob
IDX={}
for f in sorted(glob.glob(D_DIR+'*.gguf')):
    r=GGUFReader(f,'r')
    for t in r.tensors:
        IDX[t.name]=(f,int(t.data_offset),int(t.tensor_type))
    del r
TYPE2NAME={0:'f32',13:'q5_k',29:'iq2_xxs',30:'bf16',8:'q8_0'}
_files={}
def tget(name):
    path,off,ty=IDX[name]
    return path,off,ty
_QBLK={8:(34,32),10:(84,256),11:(112,256),12:(144,256),13:(176,256),14:(144,128),29:(84,256),3:(18,32)}
_QCLS={}
def dq(name,rows,cols):
    path,off,ty=tget(name)
    if ty in _QBLK:
        b,blk=_QBLK[ty]; nb=(cols+blk-1)//blk; need=rows*nb*b
    elif ty==0: need=rows*cols*4
    elif ty in (1,2): need=rows*cols*2
    else: raise ValueError('unsupported type %d'%ty)
    with open(path,'rb') as f:
        f.seek(off); raw=np.frombuffer(f.read(need),dtype=np.uint8)
    if ty==0:
        return np.frombuffer(raw,dtype=np.float32).reshape(rows,cols).copy()
    if ty==2:
        u=raw.view(np.uint16).astype(np.uint32)
        return ((u<<16).view(np.float32)).reshape(rows,cols).copy()
    import gguf.quants as _q
    enum={8:'Q8_0',12:'Q4_K',13:'Q5_K',10:'Q2_K',11:'Q3_K',29:'IQ2_XXS',14:'IQ3_XXS',3:'Q4_0'}[ty]
    rv=raw.reshape(rows,(cols+blk-1)//blk*b)
    a=np.asarray(_q.dequantize(rv,getattr(_q.GGMLQuantizationType,enum)))
    return np.ascontiguousarray(a.astype(np.float32))
def rmsn(v,eps=1e-20): return v/np.sqrt((v*v).mean(axis=-1,keepdims=True)+eps)
def rope(v,pos):
    nn=v.shape[-1]; ang=pos/10000.0**(np.arange(0,nn,2)/nn)
    c,s=np.cos(ang),np.sin(ang); out=v.copy()
    out[...,0::2]=v[...,0::2]*c-v[...,1::2]*s; out[...,1::2]=v[...,1::2]*c+v[...,0::2]*s
    return out
def e4m3(v):
    s=np.sign(v); a=np.clip(np.abs(v),0,448.0)
    e=np.clip(np.floor(np.log2(np.maximum(a,2**-6))),-6,8)
    step=np.ldexp(1.0,(e-3.0).astype(np.int32))
    qn2=np.minimum(np.round(a/step)*step,448.0)
    qs=np.round(a/2**-9)*2**-9
    return s*np.where(a>=2**-6,qn2,qs)
L=int(sys.argv[1]) if len(sys.argv)>1 else 1
P=int(sys.argv[2]) if len(sys.argv)>2 else 65
n=P+1
x=np.fromfile(f'/tmp/hg_xn_L{L:02d}.bin',dtype=np.float32)   # (n,5120) normed attention inputs ALL rows
assert x.size==n*5120, x.size
x=x.reshape(n,5120)
b=f'blk.{L}.'
kvw=dq(b+'attn_kv.weight',512,5120); kvn=dq(b+'attn_kv_a_norm.weight',1,512)[0]
qa=dq(b+'attn_q_a.weight',1280,5120);  qn=dq(b+'attn_q_a_norm.weight',1,1280)[0]
qb=dq(b+'attn_q_b.weight',32768,1280)
KV=rmsn(x@kvw.T)*kvn
qr=rmsn(x@qa.T)*qn   # per-row RMSNorm (v3's global form was OK only for 1 row)
q=(qr@qb.T).reshape(n,64,512)[P]
q=rmsn(q)
Qp=q.copy(); Qp[:,448:]=rope(q[:,448:],P)
KVp=KV.copy()
for t in range(n): KVp[t,448:]=rope(KV[t,448:][None,:],t)[0]
KVf=KVp.copy()
blk=KVf[:,:448].reshape(n,7,64)
sf=np.maximum(np.abs(blk).max(axis=2,keepdims=True)/448.0,1e-12)
KVf[:,:448]=(e4m3(blk/sf)*sf).reshape(n,448)
sink=dq(b+'attn_sinks.weight',1,64)[0]
o=np.empty((64,512),np.float32); scl=1/np.sqrt(512)
for hd in range(64):
    scs=KVf@Qp[hd]*scl
    m=max(scs.max(),sink[hd])
    pe=np.exp(scs-m); d=pe.sum()+np.exp(sink[hd]-m)
    o[hd]=(pe@KVf)/d
ho=None
import os as _os
_ph=f'/tmp/hb_ho_L{L:02d}.bin'
if _os.path.exists(_ph):
    ho=np.fromfile(_ph,dtype=np.float32).reshape(64,512)
    print(f'L{L} raw o : maxerr %.5f corr %.5f rms g %.4f rt %.4f'%(np.abs(o-ho).max(),np.corrcoef(o.ravel(),ho.ravel())[0,1],(o**2).mean()**.5,(ho**2).mean()**.5))
else:
    print(f'L{L} raw o : (no dump) golden rms %.4f'%((o**2).mean()**.5))
o3=o.copy(); o3[:,448:]=rope(o[:,448:],-P)
woa=dq(b+'attn_output_a.weight',8192,4096).reshape(8,1024,4096)
og=np.einsum('grd,gd->gr',woa,o3.reshape(8,4096)).ravel()
wob=dq(b+'attn_output_b.weight',5120,8192)
ya=wob@og
ay=np.fromfile(f'/tmp/hb_ay_L{L:02d}.bin',dtype=np.float32)
print('ya    : maxerr %.5f  corr %.5f  rms g %.4f rt %.4f'%(np.abs(ya-ay).max(),np.corrcoef(ya,ay)[0,1],(ya**2).mean()**.5,(ay**2).mean()**.5))
