# Golden L0 attention chain for ARBITRARY prompt length (generalizes g0_chain_v2
# beyond CAP): gathers per-token embedding rows from shard2 and per-row comb
# weights from /tmp/hb_ca_full_L00.bin. Compares row P (last) vs hb_*_L00 dumps.
import numpy as np, subprocess, os, tempfile, sys
SHARD2='/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00002-of-00006.gguf'
GD=os.path.expanduser('~/projects/personal/deepseek41-gufo-extension/debug/gdump2')
EMB_BASE=455105216; EMB_STRIDE=3520
_F=open(SHARD2,'rb')
def dq(off,nb,typ,rows,cols):
    _F.seek(off); buf=_F.read(nb)
    assert len(buf)==nb,(off,len(buf),nb)
    tmp=tempfile.mktemp(suffix='.bin'); open(tmp,'wb').write(buf)
    r=subprocess.run([GD,tmp,typ,str(rows),str(cols)],capture_output=True)
    os.unlink(tmp)
    if r.returncode: raise RuntimeError(f'gdump2 fail {off} {typ} {rows}x{cols}: {r.stderr[:200]}')
    a=np.frombuffer(r.stdout,dtype=np.float32)
    assert a.size==rows*cols
    return a.reshape(rows,cols).copy()
_ec={}
def emb(i):
    if i not in _ec: _ec[i]=dq(EMB_BASE+i*EMB_STRIDE,EMB_STRIDE,'q5_k',1,5120)[0]
    return _ec[i]
def rmsn(v,eps=1e-20):
    return v/np.sqrt((v*v).mean(axis=-1,keepdims=True)+eps)

tokens=np.fromfile('/tmp/hb_tokens.bin',dtype=np.int32); n=len(tokens); P=n-1
ca_full=np.fromfile('/tmp/hb_ca_full_L00.bin',dtype=np.float32)
assert ca_full.size==n*24,(ca_full.size,n*24)
ca=ca_full.reshape(n,24)
E=np.stack([emb(int(t)) for t in tokens])          # (n,5120)
h=E*ca[:,:4].sum(axis=1)[:,None]
an=dq(912958144,20480,'f32',1,5120)[0]
x=h/np.sqrt((h*h).mean()+1e-20)*an                 # global norm (matches runtime)
qa=dq(964883136,1280*5440,'q8_0',1280,5120)
qr=(qa@x.T)
qn=dq(971846336,1280*4,'f32',1,1280)[0]
qr=qr/np.sqrt((qr*qr).mean()+1e-20)*qn[:,None]
qb=dq(971851456,32768*880,'q5_k',32768,1280)
q=(qb@qr).T.reshape(n,64,512)[P]
kvw=dq(910170816,512*5440,'q8_0',512,5120)
kvn=dq(912956096,2048,'f32',1,512)[0]
KV=(kvw@x.T).T
KV=rmsn(KV)*kvn
q=rmsn(q)
RD=64; th=10000.0
def rope(v,pos):
    nn=v.shape[-1]; ang=pos/th**(np.arange(0,nn,2)/nn)
    c,s=np.cos(ang),np.sin(ang); out=v.copy()
    out[...,0::2]=v[...,0::2]*c-v[...,1::2]*s; out[...,1::2]=v[...,1::2]*c+v[...,0::2]*s
    return out
Qp=q.copy(); Qp[:,448:]=rope(q[:,448:],P)
KVp=KV.copy()
for t in range(n): KVp[t,448:]=rope(KV[t,448:][None,:],t)[0]
def e4m3(v):
    s=np.sign(v); a=np.clip(np.abs(v),0,448.0)
    e=np.clip(np.floor(np.log2(np.maximum(a,2**-6))),-6,8)
    step=np.ldexp(1.0,(e-3.0).astype(np.int32))
    qn2=np.minimum(np.round(a/step)*step,448.0)
    qs=np.round(a/2**-9)*2**-9
    return s*np.where(a>=2**-6,qn2,qs)
KVf=KVp.copy()
blk=KVf[:,:448].reshape(n,7,64)
sf=np.maximum(np.abs(blk).max(axis=2,keepdims=True)/448.0,1e-12)
KVf[:,:448]=(e4m3(blk/sf)*sf).reshape(n,448)
sink=dq(1000687296,256,'f32',1,64)[0]
o=np.empty((64,512),np.float32); sc=1/np.sqrt(512)
for hd in range(64):
    scs=KVf@Qp[hd]*sc
    m=max(scs.max(),sink[hd])
    pe=np.exp(scs-m); d=pe.sum()+np.exp(sink[hd]-m)
    o[hd]=(pe@KVf)/d
ho=np.fromfile('/tmp/hb_ho_L00.bin',dtype=np.float32).reshape(64,512)
print('raw o : maxerr %.5f  corr %.5f  rms g %.4f rt %.4f'%(np.abs(o-ho).max(),np.corrcoef(o.ravel(),ho.ravel())[0,1],(o**2).mean()**.5,(ho**2).mean()**.5))
o3=o.copy(); o3[:,448:]=rope(o[:,448:],-P)
hr=np.fromfile('/tmp/hb_hr_L00.bin',dtype=np.float32).reshape(64,512)
print('hr    : maxerr %.5f  corr %.5f'%(np.abs(o3-hr).max(),np.corrcoef(o3.ravel(),hr)[0,1] if False else np.corrcoef(o3.ravel(),hr.ravel())[0,1]))
woa=dq(912978624,8192*2816,'q5_k',8192,4096).reshape(8,1024,4096)
og=np.einsum('grd,gd->gr',woa,o3.reshape(8,4096)).ravel()
ol=np.fromfile('/tmp/hb_ol_L00.bin',dtype=np.float32)
print('ol    : maxerr %.5f  corr %.5f'%(np.abs(og-ol).max(),np.corrcoef(og,ol)[0,1]))
wob=dq(936047296,5120*5632,'q5_k',5120,8192)
ya=wob@og
ay=np.fromfile('/tmp/hb_ay_L00.bin',dtype=np.float32)
print('ya    : maxerr %.5f  corr %.5f  rms g %.4f rt %.4f'%(np.abs(ya-ay).max(),np.corrcoef(ya,ay)[0,1],(ya**2).mean()**.5,(ay**2).mean()**.5))
