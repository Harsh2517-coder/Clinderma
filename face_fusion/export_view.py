"""Writes a self-contained HTML page with the 3D face map (opens offline in any browser). Also used to build the JSON the app serves.
    from face_fusion.export_view import face3d_data, write_html"""
import base64, json, os
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def face3d_data(m, result):
    """JSON for static/face3d.js: the shared mesh, the unique spots on it (position, zone, how many photos saw them)."""
    return {"vertices": np.round(m["C"], 4).ravel().tolist(), "tris": m["tris"].ravel().tolist(), "uv": np.round(m["uv"], 4).ravel().tolist(),
            "spots": [{"p": [round(c, 4) for c in s["pos"]], "uv": [round(c, 4) for c in s["uv"]], "zone": s["zone"], "n": s["n_views"]} for s in result["spots"] if s["pos"]]}


def write_html(path, data, atlas_bgr, title="Face map"):
    jpg = cv2.imencode(".jpg", atlas_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(); three = open(os.path.join(ROOT, "static", "vendor", "three.min.js")).read(); js = open(os.path.join(ROOT, "static", "face3d.js")).read()
    html = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{margin:0;font:14px system-ui;background:#f6f4ef;color:#222}}#w{{max-width:640px;margin:0 auto;padding:12px}}.k i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px}}</style>
<div id=w><b>{title}</b><div class=k><i style="background:#1fa35c"></i>seen in 2+ photos<i style="background:#f0a020"></i>seen in 1 photo &nbsp; <label><input type=checkbox id=t checked> show spots</label></div><div id=v></div>
<small>Drag to turn, scroll to zoom, double-click to reset.</small></div><script>{three}</script><script>{js}</script>
<script>const D={json.dumps(data)};const h=FaceVeda3D.mount(document.getElementById('v'),D,'data:image/jpeg;base64,{base64.b64encode(jpg).decode()}');document.getElementById('t').onchange=e=>h.setSpotsVisible(e.target.checked)</script>"""
    open(path, "w").write(html)
