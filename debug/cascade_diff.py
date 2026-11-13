#!/usr/bin/env python3
"""Compare per-layer stream stats between a coherent run (/tmp/hb_run1) and a
collapsed run (/tmp/hb_rep50). Locates the FIRST layer whose per-token
activation diversity collapses (all rows identical / rms blowup / dead slots).
Usage: debug/cascade_diff.py [dirA] [dirB]
"""
import sys, os
import numpy as np

A = sys.argv[1] if len(sys.argv) > 1 else '/tmp/hb_run1'
B = sys.argv[2] if len(sys.argv) > 2 else '/tmp/hb_rep50'
HC, D = 4, 5120

def stats(d, L):
    f = f'{d}/hb_all_in_{L:02d}.bin'
    if not os.path.exists(f): return None
    n = os.path.getsize(f) // (4*4*D)
    x = np.fromfile(f, dtype=np.float32)[:n*4*D].reshape(n, 4, D)
    rows = x.reshape(n, -1)
    rms = float(np.sqrt((rows*rows).mean()))
    # token diversity: mean pairwise distance / rms (0 => all rows identical)
    r = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-30)
    g = r @ r.T
    div = float(1.0 - (g.sum() - np.trace(g)) / max(1, n*(n-1)))
    slot_rms = [float(np.sqrt((x[:,h,:]*x[:,h,:]).mean())) for h in range(4)]
    # per-position stream-3 (residual) growth
    s3 = float(np.sqrt((x[:,3,:]*x[:,3,:]).mean()))
    return n, rms, div, slot_rms, s3

print(f'{"L":>3} {"A.n":>4} {"A.rms":>8} {"A.div":>7} | {"B.n":>4} {"B.rms":>8} {"B.div":>7} | {"A.s3":>8} {"B.s3":>8}')
for L in range(40):
    sa, sb = stats(A, L), stats(B, L)
    if sa is None or sb is None: continue
    (na, ra, da, sl, r3), (nb, rb, db, sl2, s3b) = sa, sb
    print(f'{L:>3} {na:>4} {ra:>8.3f} {da:>7.4f} | {nb:>4} {rb:>8.3f} {db:>7.4f} | {r3:>8.3f} {s3b:>8.3f}')
