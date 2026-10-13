"""Screenshots of the built pages, and the site's social preview image.

Serves ``site/`` and ``outputs/`` on local ports, opens the route finder and
the explorer in headless Chromium, waits for them to route, and writes:

* ``outputs/screenshot_route_finder.png`` -- the route finder on its default
  trip;
* ``site/preview.jpg`` -- the 1200x630 social card the page's Open Graph
  tags point at;
* ``outputs/screenshot_interactive.png`` -- the explorer;
* ``outputs/screenshot_warped.png`` -- the explorer with the landmass of its
  opening trip redrawn by climbing cost.

Usage: python scripts/screenshots.py   (after ``python -m hk_flat_routes map``)

Adapted from the Zurich port of flattensf (MoritzFS/flatten_zrh, MIT).
"""
from __future__ import annotations

import functools
import http.server
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hk_flat_routes.config import OUTPUT_DIR, SITE_DIR  # noqa: E402


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def main() -> int:
    from PIL import Image
    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    site, port = serve(SITE_DIR)
    outs, oport = serve(OUTPUT_DIR)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()

        def page(w, h, scale=1):
            p = browser.new_page(viewport={"width": w, "height": h},
                                 device_scale_factor=scale)
            p.on("pageerror", lambda e: errors.append(str(e)))
            return p

        # the route finder, on the trip it opens with
        p = page(1440, 900, 2)
        p.goto(f"http://127.0.0.1:{port}/index.html")
        p.wait_for_function("window.App && App.family && !App.family.partial",
                            timeout=180_000)
        p.wait_for_timeout(1500)
        shot = OUTPUT_DIR / "screenshot_route_finder.png"
        p.screenshot(path=str(shot))
        print("wrote", shot)

        # the social card: 1200x630, the same page in a card-shaped window
        p = page(1200, 630, 1)
        p.goto(f"http://127.0.0.1:{port}/index.html")
        p.wait_for_function("window.App && App.family && !App.family.partial",
                            timeout=180_000)
        p.wait_for_timeout(1500)
        tmp = SITE_DIR / "preview.png"
        p.screenshot(path=str(tmp))
        Image.open(tmp).convert("RGB").save(SITE_DIR / "preview.jpg", "JPEG",
                                            quality=86, optimize=True)
        tmp.unlink()
        print("wrote", SITE_DIR / "preview.jpg")

        # the explorer, if it has been built
        explorer = OUTPUT_DIR / "hk_flat_routes_map.html"
        if explorer.exists():
            p = page(1440, 900, 2)
            p.goto(f"http://127.0.0.1:{oport}/{explorer.name}")
            p.wait_for_timeout(15_000)
            shot = OUTPUT_DIR / "screenshot_interactive.png"
            p.screenshot(path=str(shot))
            print("wrote", shot)

            # the warped city: the same page with the warp switched on
            p.click("#warptoggle")
            p.wait_for_function(
                "App.warp && !App._warping"
                " && !document.getElementById('busy').classList.contains('on')",
                timeout=180_000)
            p.wait_for_timeout(2500)
            shot = OUTPUT_DIR / "screenshot_warped.png"
            p.screenshot(path=str(shot))
            print("wrote", shot)
        browser.close()
    site.shutdown(); outs.shutdown()
    if errors:
        print("page errors:", *errors[:5], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
