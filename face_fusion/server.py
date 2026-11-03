"""Standalone local page: upload 2-3 photos of one face, get the 3D face map with the spots placed on it. No login, nothing is stored.
    python3 -m face_fusion.server            then open http://localhost:7891     (PORT=... to change)"""
import base64, json, os, sys
import cv2, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_fusion import atlas, export_view, fuse as F, pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); app = FastAPI()
PAGE = """<!doctype html><meta charset=utf-8><title>3D face map</title><style>body{font:15px system-ui;max-width:980px;margin:20px auto;padding:0 14px}
.row{display:flex;gap:10px;flex-wrap:wrap}.row label{flex:1;min-width:200px;border:1px dashed #999;padding:10px;border-radius:8px}button{padding:10px 18px;font-size:15px}
#msg{margin:12px 0;color:#444}#v,#flat{min-height:600px}</style>
<h2>3D face map</h2><p>Upload 2-3 photos of the same face (front, left side, right side). The spots are found in each photo, placed on one 3D face and merged when two photos show the same spot.</p>
<div class=row><label>Photo 1<br><input type=file accept="image/*" id=f1></label><label>Photo 2<br><input type=file accept="image/*" id=f2></label><label>Photo 3 (optional)<br><input type=file accept="image/*" id=f3></label></div>
<p><label><input type=checkbox id=det checked> find spots (uncheck to only build the 3D face)</label></p><button id=go>Build 3D map</button><div id=msg></div><div id=v></div><h3>Flat texture (full resolution: drag to move, scroll to zoom)</h3><div id=flat></div><p><a id=dl download="face_texture.jpg">Download the full-resolution texture</a></p><pre id=sum style="white-space:pre-wrap"></pre>
<script src="/three.js"></script><script src="/face3d.js"></script><script>
const rd=f=>new Promise(r=>{const x=new FileReader();x.onload=()=>r(x.result);x.readAsDataURL(f)});
document.getElementById('go').onclick=async()=>{const msg=document.getElementById('msg'),fs=['f1','f2','f3'].map(i=>document.getElementById(i).files[0]).filter(Boolean);
 if(fs.length<2){msg.textContent='Choose at least 2 photos.';return}
 msg.textContent='Working... (about 10-30 seconds)';document.getElementById('v').innerHTML='';
 const body={photos:await Promise.all(fs.map(async f=>({name:f.name,data:await rd(f)}))),detect:document.getElementById('det').checked};
 const r=await fetch('/api/fuse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();
 if(!r.ok){msg.textContent='Error: '+(j.detail||r.status);return}
 msg.textContent='Done. Drag to turn, scroll to zoom, double-click to reset. Green dots = seen in 2+ photos, amber = seen in 1.';
 FaceVeda3D.mount(document.getElementById('v'),j.data,j.atlas);document.getElementById('flat').innerHTML='';FaceVeda3D.mountFlat(document.getElementById('flat'),j.data,j.atlas);document.getElementById('dl').href=j.atlas;document.getElementById('sum').textContent=JSON.stringify(j.summary,null,1)};
</script>"""


class Req(BaseModel):
    photos: list; detect: bool = True


@app.get("/", response_class=HTMLResponse)
def page(): return PAGE


@app.get("/three.js")
def three(): return FileResponse(os.path.join(ROOT, "static", "vendor", "three.min.js"), media_type="text/javascript")


@app.get("/face3d.js")
def face3d(): return FileResponse(os.path.join(ROOT, "static", "face3d.js"), media_type="text/javascript")


@app.post("/api/fuse")
def fuse_api(b: Req):
    if not 2 <= len(b.photos) <= 6: raise HTTPException(422, "Send 2 to 6 photos")
    images = {}
    for i, p in enumerate(b.photos):
        im = cv2.imdecode(np.frombuffer(base64.b64decode(p["data"].split(",", 1)[-1]), np.uint8), cv2.IMREAD_COLOR)
        if im is None: raise HTTPException(422, f"Photo {i + 1} could not be read")
        images[f"{i + 1}: {p.get('name', 'photo')}"] = im
    spots = {n: [] for n in images}
    if b.detect:
        import scan
        for n, im in images.items():
            s = scan.scan_image(im); spots[n] = s["spots"] if s["status"] == "ok" else []
    out = pipeline.fuse_photos(images, spots); res = out["result"]
    if not res or len(out["views"]) < 2: raise HTTPException(422, "A face mesh could be fitted in fewer than 2 photos (90-degree profiles and close-ups cannot be used). Fitted: " + ", ".join(v.name for v in out["views"]))
    at, _, _ = atlas.stitch(out["model"], out["views"], images, int(os.environ.get("ATLAS_SIZE", 3072))); jpg = cv2.imencode(".jpg", at, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tobytes()
    summary = {"photos_used": [v.name for v in out["views"]], "head_angles_deg": {v["name"]: v["yaw"] for v in res["views"]}, "photos_not_usable": out["skipped"], "spots_in_photos": res["raw_total"],
               "unique_spots": res["unique_total"], "merged_duplicates": res["merged"], "spots_per_zone": res["zone_counts"], "area_seen_by_one_photo": {z: round(v, 2) for z, v in res["coverage_one_view"].items()},
               "area_seen_by_two_photos": {z: round(v, 2) for z, v in res["coverage_two_views"].items()}}
    return {"data": export_view.face3d_data(out["model"], res), "atlas": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(), "summary": summary}


if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 7891)))
