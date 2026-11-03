"""Command line: several photos of one face -> stitched face texture, 3D viewer page, and the unique spots.
    python3 -m face_fusion.cli photo1.jpg photo2.jpg photo3.jpg --out out/ [--spots spots.json] [--radius 0.04]
Spots come from the platform's scanner (V_3 + trained filter) unless --spots is given: a JSON {"photo1.jpg": [{"x": px, "y": px}, ...], ...} in each photo's own pixels
(e.g. the centres of your own detector's boxes). Writes atlas.jpg, face3d.html (opens offline), fused_spots.json."""
import argparse, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_fusion import atlas, export_view, fuse as F, pipeline


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("photos", nargs="+"); ap.add_argument("--out", default="fusion_out"); ap.add_argument("--spots"); ap.add_argument("--radius", type=float, default=F.MATCH_RADIUS)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); images = {os.path.basename(p): cv2.imread(p) for p in a.photos}
    if a.spots: spots = json.load(open(a.spots))
    else:
        import scan; spots = {}
        for n, im in images.items():
            s = scan.scan_image(im); spots[n] = s["spots"] if s["status"] == "ok" else []; print(n, len(spots[n]), "spots" if s["status"] == "ok" else "(" + s["message"] + ")")
    out = pipeline.fuse_photos(images, spots, a.radius); res = out["result"]
    if not res: sys.exit("No photo had a face mesh that could be fitted (profile photos at 90 degrees and close-ups cannot be used).")
    at, _, _ = atlas.stitch(out["model"], out["views"], images); cv2.imwrite(os.path.join(a.out, "atlas.jpg"), at)
    export_view.write_html(os.path.join(a.out, "face3d.html"), export_view.face3d_data(out["model"], res), at, "3D face map")
    json.dump({k: v for k, v in res.items()}, open(os.path.join(a.out, "fused_spots.json"), "w"), indent=1)
    print(f"{res['raw_total']} spots in {len(out['views'])} photos -> {res['unique_total']} unique ({res['merged']} merged). Zones: {res['zone_counts']}. Skipped photos: {out['skipped']}. Wrote {a.out}/")


if __name__ == "__main__": main()
