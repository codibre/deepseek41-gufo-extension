import sys, subprocess, time, requests
BIN='/home/tosol/projects/personal/deepseek41-gufo-extension/build-gufo/build/release/gufo'
base='The quick brown fox jumps over the lazy dog. '
CASES={
 'CAP':'The capital of France is',
 'F':'The secret passcode is 7391-XY. The passcode is',
 'P2':'One two three four five six seven eight nine ten eleven twelve. The number after twelve is',
 'C26':base*2+'\nThe secret passcode is 7391-XY. The passcode is',
 'C17':base*1+'\nThe secret passcode is 7391-XY. The passcode is',
 'P20':'Q: sky blue. R: grass green. Q: sun bright. R:',
 'P40':'Q: red rose. R: green leaf. Q: white snow. R: black coal. Q: gold ring. R:',
 'RPT':'a b c d e f g h i j k l m n o p q r s t u v w x y z. The letter after z is',
 'C20':'The quick brown fox jumps over the lazy dog. The secret code word is pineapple. The code word is',
 'C29':'In a small town named Brightwater, a librarian called Emma kept the secret code word pineapple. The code word is',
 'C51':'The quick brown fox jumps over the lazy dog. '*3+'\nThe secret passcode is 7391-XY. The passcode is',
 'C69':'The quick brown fox jumps over the lazy dog. '*4+'\nThe secret passcode is 7391-XY. The passcode is',
 'C111':'The quick brown fox jumps over the lazy dog. '*7+'\nThe secret passcode is 7391-XY. The passcode is',
 'C153':'The quick brown fox jumps over the lazy dog. '*10+'\nThe secret passcode is 7391-XY. The passcode is',
}
for name in sys.argv[1:]:
    p=CASES[name]
    for _ in range(3):
        try:
            if requests.get('http://127.0.0.1:8642/health',timeout=200).status_code==200: break
        except Exception: pass
        time.sleep(10)
    out=subprocess.run(['pgrep','-x','gufo'],capture_output=True).stdout.split()
    for q in out:
        try:
            rp=subprocess.check_output(['readlink',f'/proc/{q.decode()}/exe']).decode().strip()
            if rp==BIN: subprocess.run(['kill',q.decode()])
        except Exception: pass
    time.sleep(4)
    t0=time.time()
    r=requests.post('http://127.0.0.1:8642/v1/completions',timeout=200,
        json={'model':'deepseek41','prompt':p,'max_tokens':8,'temperature':0.0})
    txt=r.json()['choices'][0]['text']
    print('RES', name, repr(txt[:34]), f'{time.time()-t0:.0f}s', flush=True)
print('ISO2DONE')
