# Host analysis of the wo_a micro-probe (/tmp/gw_*): separates wrong weight
# image (map vs GGUF) from wrong contraction (device low vs numpy einsum).
import numpy as np, subprocess, os, tempfile

SHARD2='/home/tosol/.lmstudio/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00002-of-00006.gguf'
GD=os.path.expanduser('~/.hermes/cache/scratch/gdump2')
OFF_WOA=912978624; ROWS=8192; COLS=4096; NB=ROWS*COLS*176//256
RANK=1024; GROUPS=8; TOKS=4

def dq_file(path, off, nb, typ, rows, cols):
    with open(path,'rb') as f:
        f.seek(off); buf=f.read(nb)
    tmp=tempfile.mktemp(suffix='.bin'); open(tmp,'wb').write(buf)
    r=subprocess.run([GD,tmp,typ,str(rows),str(cols)],capture_output=True,check=True)
    os.unlink(tmp)
    a=np.frombuffer(r.stdout,dtype=np.float32)
    assert a.size==rows*cols,(a.size,rows*cols)
    return a.reshape(rows,cols).copy()

Wg = dq_file(SHARD2, OFF_WOA, NB, 'q5_k', ROWS, COLS)          # golden weights
Wm = dq_file('/tmp/gw_map.bin', 0, NB, 'q5_k', ROWS, COLS)     # runtime-mapped weights
print('map size', os.path.getsize('/tmp/gw_map.bin'), 'expect', NB)
print('W vs GGUF: equal bytes?', np.array_equal(Wg, Wm),
      'max|diff|', float(np.abs(Wg-Wm).max()))

x = np.fromfile('/tmp/gw_heads.bin', dtype=np.float32).reshape(TOKS, GROUPS, COLS)
xl= np.fromfile('/tmp/gw_low.bin', dtype=np.float32).reshape(TOKS, GROUPS, RANK)
# x[t,g,d] = ((d*13 + t*7 + g) % 29 - 14) * 0.25
d=np.arange(COLS); 
for t in range(TOKS):
    for g in range(GROUPS):
        exp=(( (d*13 + t*7 + g) % 29) -14)*0.25
        assert np.allclose(x[t,g],exp), f'x mismatch t{t} g{g}'
print('x verified against probe formula')

exp_low=np.einsum('grd,tgd->tgr', Wg.reshape(GROUPS,RANK,COLS), x)  # (4,8,1024)
err=np.abs(exp_low-xl)
print('low: max err', float(err.max()),
      'rms exp', float((exp_low**2).mean()**.5),
      'rms got', float((xl**2).mean()**.5))
for g in range(GROUPS):
    ge=err[:,g]; gr_exp=(exp_low[:,g]**2).mean()**.5; gr_got=(xl[:,g]**2).mean()**.5
    corr=float(np.corrcoef(exp_low[:,g].ravel(), xl[:,g].ravel())[0,1]) if gr_exp>0 and gr_got>0 else 0.0
    print(f'g{g}: rms exp {gr_exp:.4f} got {gr_got:.4f} maxerr {ge.max():.4f} corr {corr:.4f}')
