"""Record every loop transition on the route page and look for hard cuts in
the map: a cut is one frame whose map pixels change far more than its
neighbours'.

    python tests/qa_map_cuts.py [SITE_DIR] [TAG]

A check to run by hand, not a test (pytest does not collect it).  It serves
the built site (``config.SITE_DIR`` unless a directory is given, so a run
redirected with ``HKFR_RUN_DIR`` checks its own site), records the page in
headless Chromium while it is put through loop mode at desktop and phone
sizes, splits the recording into frames and measures how much of the map
changes from each frame to the next.  A glide spreads the change over
several frames; a cut puts it all in one.

Needs Playwright with Chromium, Pillow, and an ffmpeg to split the
recording: one on the PATH, the one ``imageio-ffmpeg`` ships, or the one
Playwright downloads for its own recordings.
"""
import functools, glob, http.server, os, shutil, subprocess, sys, tempfile, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

from hk_flat_routes.config import SITE_DIR

ROOT = sys.argv[1] if len(sys.argv) > 1 else str(SITE_DIR)
SP = os.path.join(tempfile.gettempdir(), "hkfr_map_cuts_" + (sys.argv[2] if len(sys.argv) > 2 else "a"))
#: Loop lengths to slide between, in the page's default unit (kilometres).
LONG_LOOP, SHORT_LOOP = 8, 3


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe: return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    # Playwright keeps a small ffmpeg beside its browsers for recording
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "~/Library/Caches/ms-playwright",
             "~/.cache/ms-playwright", os.path.join(os.environ.get("LOCALAPPDATA", ""), "ms-playwright")]
    for root in roots:
        hits = sorted(glob.glob(os.path.join(os.path.expanduser(root), "ffmpeg-*", "ffmpeg-*"))) if root else []
        if hits: return hits[-1]
    sys.exit("no ffmpeg found: put one on the PATH, or `pip install imageio-ffmpeg`")


if not os.path.exists(os.path.join(ROOT, "index.html")):
    sys.exit(f"no site at {ROOT}: build it with `python -m hk_flat_routes map`")
FF = find_ffmpeg()


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a, **k): pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=ROOT))
threading.Thread(target=srv.serve_forever, daemon=True).start()


def run(vp, label):
    shutil.rmtree(SP, ignore_errors=True); os.makedirs(SP)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        ctx = b.new_context(viewport=vp, record_video_dir=SP, record_video_size=vp, is_mobile=vp["width"] < 600, has_touch=vp["width"] < 600)
        page = ctx.new_page()
        page.goto(f"http://127.0.0.1:{srv.server_address[1]}/", wait_until="load")
        settle = lambda: (page.wait_for_function("window.App && App.family && !App.family.partial && !App._scan", timeout=120000), page.wait_for_timeout(1200))
        settle(); page.evaluate("""() => { window._ev = []; const m = document.createElement('div'); m.id = 'qamark';
          Object.assign(m.style, { position: 'fixed', right: '0', top: '0', width: '24px', height: '24px', background: '#000', zIndex: 99999, display: 'none' }); document.body.appendChild(m); }""")
        def mark(name):
            page.evaluate(f"() => {{ _ev.push('{name}'); const m = document.getElementById('qamark'); m.style.display = 'block'; setTimeout(() => m.style.display = 'none', 160); }}")
            page.wait_for_timeout(220)
        def slide(v):
            page.evaluate(f"() => {{ const s = document.getElementById('sl'); s.value = {v}; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }}")
        mark("loop on"); page.click("#loopbtn"); settle()
        mark(f"slider {LONG_LOOP}"); slide(LONG_LOOP); settle()
        for k in range(2):
            if page.locator("#nextloop").count():
                mark("another loop"); page.click("#nextloop"); page.wait_for_timeout(1200)
        mark(f"slider {SHORT_LOOP}"); slide(SHORT_LOOP); settle()
        mark("out-and-back on"); page.click("#outback"); settle()
        mark("loop off"); page.click("#loopbtn"); settle()
        ev = page.evaluate("() => _ev")
        ctx.close(); b.close()
    v = glob.glob(SP + "/*.webm")[0]
    # every frame of the recording as it was captured (Playwright records at
    # a steady 25 a second), so nothing here depends on ffmpeg's filters
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", v, "-vsync", "0", f"{SP}/f%04d.png"], check=True)
    fs = sorted(glob.glob(f"{SP}/f*.png"))
    # the map region: right of the card on wide screens, above it on phones
    wide = vp["width"] > 640
    crop = (440, 0, vp["width"], vp["height"]) if wide else (0, 0, vp["width"], int(vp["height"] * 0.42))
    prev = None; d = []; marks = []
    for i, f in enumerate(fs):
        full = Image.open(f).convert("L")
        corner = np.asarray(full.crop((vp["width"] - 20, 2, vp["width"] - 4, 18))).mean()
        if corner < 60 and (not marks or i - marks[-1] > 6): marks.append(i)
        im = full.crop(crop).resize(((crop[2]-crop[0])//2, (crop[3]-crop[1])//2))
        d.append(0 if prev is None else float(np.asarray(ImageChops.difference(im, prev)).mean())); prev = im
    d = np.array(d)
    print(f"\n### {label} {vp}: {len(fs)} frames, {len(marks)} marks for {len(ev)} events")
    bad = False
    for j, name in enumerate(ev):
        if j >= len(marks): break
        f0 = marks[j] + 4; f1 = marks[j + 1] if j + 1 < len(marks) else len(d)
        win = d[f0:f1]
        if not len(win): continue
        peak = win.max(); frames = (win > 1.0).sum()
        # a glide spreads change over several frames; a cut puts it in one
        kind = "cut" if peak > 5 and frames <= 2 else ("glide" if frames >= 3 else "still")
        verdict = "  <-- CUT" if kind == "cut" else ""
        if kind == "cut": bad = True
        print(f"  {name:16s} peak {peak:5.1f}  changing frames {frames:2d}  {kind}{verdict}")
    return bad


bad = run({"width": 1280, "height": 800}, "desktop") | run({"width": 390, "height": 844}, "phone")
srv.shutdown()
shutil.rmtree(SP, ignore_errors=True)
print("\nHARD CUTS FOUND" if bad else "\nno hard cuts")
