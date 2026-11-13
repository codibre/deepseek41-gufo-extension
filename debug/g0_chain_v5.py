# Partition test at layer L with RUNTIME q/kv:
#   golden attention (e4m3 blocks + sink + causal) on rt q/kv -> o_pred
#   back-rope + golden oproj -> ya_pred ; compare vs rt ay row P.
# corr~1 => attention kernel innocent, corruption is upstream (input side).
# corr<1 => attention math itself diverges.
import numpy as np, os, sys, glob
from gguf import GGUFReader
D_DIR='/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/'
IDX={}
for f in sorted(glob.glob(D_DIR+'*.gguf')):
    r=GGUFReader(f,'r')
    for t in r.tensors: IDX[t.name]=(f,int(t.data_offset),int(t.tensor_type))
    del r
_QBLK={8:(34,32),10:(84,256),11:(112,256),12:(144,256),13:(176,256),14:(144,128),29:(84,256),3:(18,32)}
def dq(name,rows,cols):
    path,off,ty=IDX[name]
    if ty in _QBLK:
        b,blk=_QBLK[ty]; nb=(cols+blk-1)//blk; need=rows*nb*b
    elif ty==0: need=rows*cols*4
    elif ty in (1,2): need=rows*cols*2
    else: raise ValueError('unsupported type %d'%ty)
    with open(path,'rb') as f:
        f.seek(off); raw=np.frombuffer(f.read(need),dtype=np.uint8)
    if ty==0: return np.frombuffer(raw,dtype=np.float32).reshape(rows,cols).copy()
    if ty==2:
        u=raw.view(np.uint16).astype(np.uint32)
        return ((u<<16).view(np.float32)).reshape(rows,cols).copy()
    import gguf.quants as _q
    b,blk=_QBLK[ty]
    enum={8:'Q8_0',12:'Q4_K',13:'Q5_K',10:'Q2_K',11:'Q3_K',29:'IQ2_XXS',14:'IQ3_XXS',3:'Q4_0'}[ty]
    rv=raw.reshape(rows,(cols+blk-1)//blk*b)
    return np.asarray(_q.dequantize(rv,getattr(_q.GGMLQuantizationType,enum)))

L=int(sys.argv[1]); P=int(sys.argv[2]); LREAL=int(os.environ.get('LREAL',L))
n=os.path.getsize('/tmp/hb_tokens.bin')//4
q=np.fromfile(f'/tmp/hb_all_aq_{LREAL:02d}.bin',dtype=np.float32).reshape(n,64,512)
kv=np.fromfile(f'/tmp/hb_all_akv_{LREAL:02d}.bin',dtype=np.float32).reshape(n,512)
print('q rms %.4f  kv rms %.4f'%(np.sqrt((q*q).mean()),np.sqrt((kv*kv).mean())))
def e4m3(v):
    s=np.sign(v); a=np.clip(np.abs(v),0,448.0)
    e=np.clip(np.floor(np.log2(np.maximum(a,2**-6))),-6,8)
    step=np.ldexp(1.0,(e-3.0).astype(np.int32))
    qn2=np.minimum(np.round(a/step)*step,448.0)
    qs=np.round(a/2**-9)*2**-9
    return s*np.where(a>=2**-6,qn2,qs)
KVf=kv.copy()
blk=KVf[:,:448].reshape(n,7,64)
sf=np.maximum(np.abs(blk).max(axis=2,keepdims=True)/448.0,1e-12)
KVf[:,:448]=(e4m3(blk/sf)*sf).reshape(n,448)
b=f'blk.{LREAL}.'
sink=dq(b+'attn_sink.weight',1,64)[0] if b+'attn_sink.weight' in IDX else None
if sink is None:
    # v3 used fixed offset 1000687296 f32 1x64 per-layer? emulate via IDX name search
    cand=[k for k in IDX if 'sink' in k and f'blk.{L}.' in k]
    print('sink candidates:',cand[:3]); sink=dq(cand[0],1,64)[0]
sc=1/np.sqrt(512)
o=np.empty((64,512),np.float32)
for hd in range(64):
    scs=KVf[:P+1]@q[P,hd]*sc
    m=max(scs.max(),sink[hd])
    pe=np.exp(scs-m); d=pe.sum()+np.exp(sink[hd]-m)
    o[hd]=(pe@KVf[:P+1])/d
th=10000.0
def rope(v,pos):
    nn=v.shape[-1]; ang=pos/th**(np.arange(0,nn,2)/nn)
    c,s=np.cos(ang),np.sin(ang); out=v.copy()
    out[...,0::2]=v[...,0::2]*c-v[...,1::2]*s; out[...,1::2]=v[...,1::2]*c+v[...,0::2]*s
    return out
o3=o.copy(); o3[:,448:]=rope(o[:,448:],-P)
woa=dq(b+'attn_output_a.weight',8192,4096).reshape(8,1024,4096)
og=np.einsum('grd,gd->gr',woa,o3.reshape(8,4096)).ravel()
wob=dq(b+'attn_output_b.weight',5120,8192)
ya=wob@og
ay=np.fromfile(f'/tmp/hb_ay_L{LREAL:02d}.bin',dtype=np.float32).copy()
print('ya(P=%d): maxerr %.5f  corr %.5f  rms g %.4f rt %.4f'%(
    P,np.abs(ya-ay).max(),np.corrcoef(ya,ay)[0,1],(ya**2).mean()**.5,(ay**2).mean()**.5))

# ---- variant battery on rt q/kv (compare o_pred vs rt ho if dumped) ----
import os.path as _op
ho=None
for cand in (f'/tmp/hb_ho_L{L:02d}.bin',):
    if _op.exists(cand): ho=np.fromfile(cand,dtype=np.float32).reshape(64,512)
if ho is None and _op.exists(f'/tmp/hb_all_ho_{LREAL:02d}.bin'):
    ho=np.fromfile(f'/tmp/hb_all_ho_{LREAL:02d}.bin',dtype=np.float32).reshape(n,64,512)[P]
sc_all=(KVf[:P+1]@q[P].T*sc) if False else None
def variants():
    out={}
    for hd in range(64): pass
    logits=KVf[:P+1]@q[P,0]*sc  # placeholder per-head below
    # V1 sink-in-max (done above) -> o
    o1=o.copy()
    # V2 sink in denom, no max shift
    o2=np.empty_like(o1)
    # V3 e4m3 disabled (raw KVp roped)
    o3=np.empty_like(o1)
    # V4 softmax over logits incl sink row with value=0
    o4=np.empty_like(o1)
    for hd in range(64):
        s=KVf[:P+1]@q[P,hd]*sc
        m=s.max(); pe=np.exp(s-m); d=pe.sum()+np.exp(sink[hd]-m)
        o2[hd]=(pe@KVf[:P+1])/d
        s2=kv[:P+1]@q[P,hd]*sc
        m2=max(s2.max(),sink[hd]); pe2=np.exp(s2-m2); d2=pe2.sum()+np.exp(sink[hd]-m2)
        o3[hd]=(pe2@kv[:P+1])/d2
        s4=np.append(KVf[:P+1]@q[P,hd]*sc, 0.0)
        p4=np.exp(s4-s4.max()); o4[hd]=(p4[:-1]@KVf[:P+1])/p4.sum()
    return o1,o2,o3,o4
o1,o2,o3,o4=variants()
names=['V1 sink+max','V2 sink no-maxshift? (same)',None,'V3 no-e4m3','V4 sinkrow0']
for nm,ov in zip(['V1 sink+max','V3 no-e4m3','V4 sinkrow0'],[o1,o3,o4]):
    o3v=ov.copy(); o3v[:,448:]=rope(ov[:,448:],-P)
    ogv=np.einsum('grd,gd->gr',woa,o3v.reshape(8,4096)).ravel()
    yav=wob@ogv
    print('%-14s ya corr %.5f rms %.4f'%(nm,np.corrcoef(yav,ay)[0,1],(yav**2).mean()**.5))
    if ho is not None:
        print('   vs ho: corr %.5f maxerr %.5f'%(np.corrcoef(ov.ravel(),ho.ravel())[0,1],np.abs(ov-ho).max()))

# V5 exclusive causal (t<P), V6 no-sink inclusive, V7 scale 1/sqrt(576)
def mk(mask_excl, use_sink, scale):
    ov=np.empty_like(o)
    for hd in range(64):
        idx=np.arange(P) if mask_excl else np.arange(P+1)
        s=KVf[idx]@q[P,hd]*scale
        if use_sink:
            m=max(s.max(),sink[hd]); pe=np.exp(s-m); d=pe.sum()+np.exp(sink[hd]-m)
        else:
            m=s.max(); pe=np.exp(s-m); d=pe.sum()
        ov[hd]=(pe@KVf[idx])/d
    o3v=ov.copy(); o3v[:,448:]=rope(ov[:,448:],-P)
    ogv=np.einsum('grd,gd->gr',woa,o3v.reshape(8,4096)).ravel()
    yav=wob@ogv
    print('%-18s ya corr %.5f | vs ho corr %.5f'%(nm:=('excl' if mask_excl else 'incl')+('_sink' if use_sink else '_nosink')+f'_1sqrt{int(scale*scale)}',
        np.corrcoef(yav,ay)[0,1], np.corrcoef(ov.ravel(),ho.ravel())[0,1]))
mk(True,True,sc); mk(False,False,sc); mk(False,True,1/np.sqrt(576)); mk(True,False,sc)
