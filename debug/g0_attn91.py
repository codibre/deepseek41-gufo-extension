# Golden L0 attention at last prompt row vs /tmp/hb_ay_L00.bin  (batch path, n=92, SWA 128)
import numpy as np, subprocess, os, tempfile
ROOT='/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00002-of-00006.gguf'
GD='/home/tosol/.hermes/cache/scratch/gdump2'
_F=open(ROOT,'rb')
def dq(off,nb,typ,rows,cols):
    _F.seek(off); buf=_F.read(nb)
    tmp=tempfile.mktemp(suffix='.bin'); open(tmp,'wb').write(buf)
    r=subprocess.run([GD,tmp,typ,str(rows),str(cols)],capture_output=True,check=True)
    os.unlink(tmp)
    a=np.frombuffer(r.stdout,dtype=np.float32)
    assert a.size==rows*cols,(off,a.size,rows*cols)
    return a.reshape(rows,cols).copy()
def emb_row(tid):
    return dq(455105216+tid*3520,3520,'q5_k',1,5120)[0]
eps=1e-20
tokens=np.fromfile('/tmp/hb_tokens.bin',dtype=np.int32); n=len(tokens); P=n-1
ca=np.fromfile('/tmp/hb_ca_L00.bin',dtype=np.float32); pre_a,post_a=ca[0:4],ca[4:8]; comb_a=ca[8:24].reshape(4,4)
E=np.stack([emb_row(int(t)) for t in tokens])            # (n,5120); L0 streams = emb tiled
h=E[P]*pre_a.sum()                                        # collapse of tiled streams
an=dq(912958144,20480,'f32',1,5120)[0]
x=h/np.sqrt((h*h).mean()+eps)*an
qa=dq(964883136,1280*5440,'q8_0',1280,5120)                # (1280,5120)
qr=qa@x
qn=dq(971846336,1280*4,'f32',1,1280)[0]
qr=qr/np.sqrt((qr*qr).mean()+eps)*qn
qb=dq(971851456,32768*880,'q5_k',32768,1280)              # (32768,1280)
q=(qb@qr).reshape(64,512)                                 # row P query heads
kvw=dq(910170816,512*5440,'q8_0',512,5120)                 # (512,5120)
kvn=dq(912956096,2048,'f32',1,512)[0]
# per-token kv (all rows) : need each row's attn_norm x_t
xs=np.empty((n,5120),np.float32)
for t in range(n):
    ht=E[t]*pre_a.sum()
    xs[t]=ht/np.sqrt((ht*ht).mean()+eps)*an
KV=(kvw@xs.T).T                                           # (n,512)
KV=KV/np.sqrt((KV*KV).mean(axis=1,keepdims=True)+eps)*kvn
q=q/np.sqrt((q*q).mean(axis=1,keepdims=True)+eps)          # per-head q rms norm (runtime kernel)
# rope partial 64, base theta 1e4, llama interleave pairs (2i,2i+1)
RD=64; th=10000.0
def rope(v, pos):  # v (m,RD) pairs (2i,2i+1), freq base^(-2j/RD)
    n=v.shape[1]
    ang=pos/th**(np.arange(0,n,2)/n)
    c,s=np.cos(ang),np.sin(ang)
    out=v.copy()
    ev=v[:,0::2]; od=v[:,1::2]
    out[:,0::2]=ev*c-od*s; out[:,1::2]=od*c+ev*s
    return out
Qp=q.copy()
Qp[:,512-RD:]=rope(q[:,512-RD:],P)
KVp=KV.copy()
for t in range(n): KVp[t,448:]=rope(KV[t,448:][None,:],t)[0]
def e4m3(v):
    s=np.sign(v); a=np.clip(np.abs(v),0,448.0)
    e=np.clip(np.floor(np.log2(np.maximum(a,2**-6))),-6,8)
    step_n=np.ldexp(1.0,(e-3.0).astype(np.int32))
    qn=np.minimum(np.round(a/step_n)*step_n,448.0)
    qs=np.round(a/2**-9)*2**-9
    return s*np.where(a>=2**-6,qn,qs)
KVf=KVp.copy()
blk=KVf[:,:448].reshape(n,7,64)
sf=np.maximum(np.abs(blk).max(axis=2,keepdims=True)/448.0,1e-12)
KVf[:,:448]=(e4m3(blk/sf)*sf).reshape(n,448)
sink=dq(1000687296,64*4,'f32',1,64)[0]
# attention row P: keys 0..P (window 128 covers all), scale 1/sqrt(512)
o=np.empty((64,512),np.float32)
sc=1/np.sqrt(512)
for hd in range(64):
    scs=KVf@Qp[hd]*sc
    m=max(scs.max(),sink[hd])
    pe=np.exp(scs-m); d=pe.sum()+np.exp(sink[hd]-m)
    o[hd]=(pe@KVf)/d
# un-rope o
o2=o.copy(); o2[:,512-RD:]=rope(o[:,512-RD:],-P)  # inverse rotation
# fp8 block-128 act quant per row
def actq(m,blk_sz=128):
    m2=m.reshape(m.shape[0],-1,blk_sz)
    am=np.abs(m2).max(axis=2,keepdims=True)
    sf=np.maximum(am/448.0,1e-12)
    return (e4m3(m2/sf)*sf).reshape(m.shape)
o3=o2  # V4.1 q5_k path: no activation quantization
woa=dq(912978624,8192*2816,'q5_k',8192,4096)
og=np.einsum('grd,hd->gr',woa.reshape(8,1024,4096),o3.reshape(8,4096)).reshape(8192)
wob=dq(936047296,5120*5632,'q5_k',5120,8192)
ya=wob@og
import os as _os
if _os.path.exists('/tmp/hb_ho_L00.bin'):
    ho_rt=np.fromfile('/tmp/hb_ho_L00.bin',dtype=np.float32).reshape(64,512)
    print('raw o err vs runtime',float(np.abs(o-ho_rt).max()),'rms rt',float((ho_rt**2).mean()**.5))
if _os.path.exists('/tmp/hb_ol_L00.bin'):
    ol_rt=np.fromfile('/tmp/hb_ol_L00.bin',dtype=np.float32)
    print('attn_low err',float(np.abs(og-ol_rt).max()),'rms g',float((og**2).mean()**.5),'rms rt',float((ol_rt**2).mean()**.5))
rt=np.fromfile('/tmp/hb_ay_L00.bin',dtype=np.float32)
err=np.abs(ya-rt)
print('golden ya[:6]',ya[:6])
print('runtime   [:6]',rt[:6])
print('max err',err.max(),'rms g',(ya**2).mean()**.5,'rms rt',(rt**2).mean()**.5,'rel',float((err.max()/ ((rt**2).mean()**.5))) )
