# README banner and social card: a still of the Minerva site mercury background (ASCII metaballs) around the HgRT
# logotype. Reads hgrt-{light,dark}.svg next to this script (copy them from docs/assets after scripts/make_logo.py).
# Needs google-chrome and websocket-client.   python scripts/banner/make.py docs/assets
import base64, json, os, subprocess, sys, tempfile, threading, time, urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import functools, websocket

D = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, 'out')
os.makedirs(OUT, exist_ok=True)

class Q(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Q, directory=D))
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]
prof = tempfile.mkdtemp(dir=D)
ch = subprocess.Popen(['google-chrome', '--headless=new', '--remote-debugging-port=9333', '--user-data-dir=' + prof,
                       '--no-first-run', '--hide-scrollbars', 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(50):
    try:
        tabs = json.load(urllib.request.urlopen('http://127.0.0.1:9333/json')); break
    except Exception: time.sleep(.2)
ws = websocket.create_connection([t for t in tabs if t['type'] == 'page'][0]['webSocketDebuggerUrl'], timeout=120)
i = 0
def cmd(m, **p):
    global i; i += 1; ws.send(json.dumps({'id': i, 'method': m, 'params': p}))
    while True:
        r = json.loads(ws.recv())
        if r.get('id') == i: return r.get('result', {})
cmd('Page.enable'); cmd('Page.navigate', url=f'http://127.0.0.1:{port}/render.html'); time.sleep(1.5)

# Blob layouts (x, y as fractions of width/height; r as fraction of the short side)
BANNER = [[0.06, 0.20, 0.30], [0.17, 0.92, 0.24], [0.27, 0.10, 0.13], [0.80, 0.18, 0.22], [0.95, 0.72, 0.30],
          [0.70, 0.95, 0.17], [0.40, 1.05, 0.10], [0.58, -0.02, 0.10]]
SOCIAL = [[0.05, 0.16, 0.26], [0.13, 0.86, 0.22], [0.86, 0.12, 0.20], [0.96, 0.62, 0.26], [0.74, 0.97, 0.16],
          [0.30, 1.02, 0.10], [0.62, -0.03, 0.09], [0.25, 0.06, 0.08]]
LW = 420; LH = LW * 190 / 466
jobs = {
  'hgrt-banner-light.png': dict(w=1280, h=400, theme='light', bg='none', balls=BANNER, clear=[0.5, 0.5, 0.26, 0.55],
                                logo=dict(src='hgrt-light.svg', w=LW, x=(1280 - LW) / 2, y=(400 - LH) / 2)),
  'hgrt-banner-dark.png': dict(w=1280, h=400, theme='dark', bg='none', balls=BANNER, clear=[0.5, 0.5, 0.26, 0.55],
                               logo=dict(src='hgrt-dark.svg', w=LW, x=(1280 - LW) / 2, y=(400 - LH) / 2)),
  'hgrt-social.png': dict(w=1280, h=640, scale=1, theme='dark', bg='#0d1117', balls=SOCIAL, clear=[0.5, 0.56, 0.44, 0.52],
                          logo=dict(src='hgrt-dark.svg', w=460, x=(1280 - 460) / 2, y=150),
                          texts=[]),
}
for name, o in jobs.items():
    if name == 'hgrt-social.png':
        f1 = '500 30px "DejaVu Sans", Arial, sans-serif'; f2 = '400 20px "DejaVu Sans Mono", monospace'
        o['texts'] = [dict(text='The Mercurius runtime and long-context stack', font=f1, color='#e6edf3', x=0, y=440, center=True),
                      dict(text='github.com/Minerva-Laboratories/HgRT', font=f2, color='#8b949e', x=0, y=490, center=True)]
    if o.get('texts'):
        # centre each line by measuring it in the page
        for tx in o['texts']:
            wpx = cmd('Runtime.evaluate', returnByValue=True, expression=
                f"(()=>{{const c=document.createElement('canvas').getContext('2d');c.font={json.dumps(tx['font'])};return c.measureText({json.dumps(tx['text'])}).width}})()")['result']['value']
            tx['x'] = (o['w'] - wpx) / 2
    r = cmd('Runtime.evaluate', expression=f'render({json.dumps(o)})', awaitPromise=True, returnByValue=True)
    v = r.get('result', {}).get('value')
    if not v: print('fail', name, r); continue
    open(os.path.join(OUT, name), 'wb').write(base64.b64decode(v.split(',', 1)[1]))
    print(name, os.path.getsize(os.path.join(OUT, name)))
ch.terminate(); srv.shutdown()
import shutil; shutil.rmtree(prof, ignore_errors=True)
