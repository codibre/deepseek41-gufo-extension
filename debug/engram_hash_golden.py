import re, numpy as np
from gguf import GGUFReader
P='/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf'
r=GGUFReader(P)
toks=np.fromfile('/tmp/hb_tokens.bin',dtype=np.int32).astype(np.int64); n=len(toks)
mult=[int(x) for x in r.fields['deepseek41.ple.layer_multipliers'].contents()]
primes=[int(x) for x in r.fields['deepseek41.ple.head_vocab_sizes'].contents()]
offs=[int(x) for x in r.fields['deepseek41.ple.head_offsets'].contents()]
tmap=np.array(r.fields['deepseek41.engram.token_map'].contents(),dtype=np.int64)
padc=int(tmap[2]); DEAD=-1
cls=[int(c) for c in tmap[toks]]
MASK=(1<<64)-1
ng=4; hpn=8
def gold(e):
    M=mult[e*ng:(e+1)*ng]; PR=primes[e*24:(e+1)*24]; OF=offs[e*24:(e+1)*24]
    G=np.zeros((n,24),dtype=np.int64)
    for i in range(n):
        blocked=False; tk=[]
        for s in range(ng):
            src = DEAD if i<s else cls[i-s]
            blocked = blocked or (src==DEAD)
            tk.append(padc if blocked else src)
        roll=(tk[0]*M[0]) & MASK
        for o in range(1,ng):
            roll ^= (tk[o]*M[o]) & MASK
            for h in range(hpn):
                col=(o-1)*hpn+h
                G[i,col]=(roll%PR[col])+OF[col]
    return G
log=open('/tmp/eg_stderr.log','rb').read().decode('utf8','replace')
for e in (0,1):
    l0=[l for l in log.split('\n') if f'mod={e} ' in l][-1]
    raw=[int(x) for x in re.findall(r'-?\d+', l0.split('first=')[1])]
    vals=[v for v in raw if v>100000][:24]
    G=gold(e)
    flat_i=G.T.reshape(-1)  # col-major: all tokens of col0, then col1...
    flat_r=G.reshape(-1)
    print(f'mod{e} n_vals {len(vals)}')
    print('  row-major match:', list(G.reshape(-1)[:len(vals)])==vals)
    print('  col-major match:', list(flat_i[:len(vals)])==vals)
    # subset checks
    g0=list(G[:,0]); m=sum(1 for k,v in enumerate(vals) if v in g0)
    print('  vals found in ref col0:', m, '/', len(vals))
    allc=set(G.reshape(-1).tolist()); m2=sum(1 for v in vals if v in allc)
    print('  vals found anywhere in ref:', m2, '/', len(vals))
    for k in range(min(6,len(vals))):
        hits=[(i,c) for i in range(n) for c in range(24) if G[i,c]==vals[k]]
        print('   log',k,vals[k],'-> ref hits',hits[:3])
