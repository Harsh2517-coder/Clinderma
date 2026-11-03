# face_fusion: put the spots from three face photos onto one face

A spot can be visible in the front photo and hidden in a side photo, or visible in both (a cheek). Counting each photo separately counts the cheek spots twice and
cannot say which spot is which. `face_fusion` maps every photo onto **one shared 3D face model** so each spot gets a place on the face, spots seen from two angles are
merged into one, and the photos are stitched into one unwrapped face texture you can turn in 3D.

Used by FaceVeda (every scan) and reusable for Clinderma: nothing in it depends on FaceVeda except the optional `cli.py` default of using FaceVeda's scanner.

## How it works
1. **Landmarks.** MediaPipe FaceLandmarker finds 478 face points in each photo (`landmark_worker.py`). If a turned face is not found it retries on a mirrored copy.
2. **Shared face model** (`models/face_canonical.npz`, built by `build_canonical.py` from 136 frontal photos): the average 3D layout of those 478 points, the mesh triangles,
   the zone of each triangle (forehead / nose / chin / left / right cheek) and its surface area.
3. **Fit each photo** (`fuse.fit_view`): head pose of the shared model in that photo, and for each triangle how much it faces the camera.
4. **Locate spots** (`fuse.locate`): a spot's pixel falls inside one mesh triangle in the photo, so its barycentric position gives a place on the shared face.
5. **Merge** (`fuse.fuse`): spots from different photos closer than 0.04 of the face width (about 5-6 mm) are one spot (one-to-one matching, spots from the same photo never merge).
   It also reports how much of each zone every photo actually saw (`coverage_one_view`, `coverage_two_views`), so a scan that missed an area says so.
6. **Stitch** (`atlas.stitch`): each photo is warped triangle by triangle onto an unwrapped face and blended where it faces the camera. `static/face3d.js` shows it in 3D.

## Setup
The system `mediapipe 1.0` crashes at start-up on macOS (Metal), so the landmark step runs in its own Python 3.12 environment:
```bash
python3.12 -m venv ../facefusion_venv && ../facefusion_venv/bin/pip install "mediapipe==0.10.21" opencv-python numpy
```
`FACEFUSION_PYTHON=/path/to/python` points to another interpreter. `models/face_landmarker.task` is Google's MediaPipe model (Apache-2.0). Everything else needs only numpy, scipy and OpenCV.

## Use
**Simplest: a local page.** `python3 face_fusion/server.py`, open http://localhost:7891, choose 2-3 photos of one face, press Build. No login, nothing is stored. (In the Claude app the launch entry is `face-3d-map`.)

```bash
python3 -m face_fusion.cli front.jpg left.jpg right.jpg --out out/          # writes atlas.jpg, face3d.html (open it, drag to turn), fused_spots.json
python3 -m face_fusion.cli *.jpg --spots my_spots.json --out out/           # use your own detector's spots: {"front.jpg": [{"x": px, "y": px}, ...], ...}
```
```python
from face_fusion.pipeline import fuse_photos
out = fuse_photos({"front": front_bgr, "left": left_bgr, "right": right_bgr}, {"front": [{"x": 812, "y": 1400}], "left": [...], "right": [...]})
out["result"]["unique_total"], out["result"]["zone_counts"], out["result"]["coverage_one_view"], out["skipped"]
```
**Clinderma:** copy the `face_fusion/` folder, `models/face_landmarker.task` and `models/face_canonical.npz` next to your code. For each photo give the centre of every accepted box
(V_3: `(core[0]+core[2])/2 + crop_box[0]`, `(core[1]+core[3])/2 + crop_box[1]` in the ORIGINAL photo's pixels).

## What was measured (100 patient visits, 465 photos from the Clinderma patient photos)
- A face mesh could be fitted in **356 of 465 photos (77%)**. The others are true 90-degree profiles (only an ear visible) and forehead or cheek close-ups: those cannot be fused and are left out (their per-photo counts are kept).
- **Do spots land in the same place from two angles?** Share of spots in the overlap of two angles that find a partner, same patient against different patients (chance), at radius 0.04:

  | spots | same patient | chance |
  |---|---|---|
  | expert-labelled lesions (46 patients, 112 angle pairs) | 13.3% | 3.9% |
  | detector output (67 visits, 165 angle pairs) | 12.5% | 4.5% |

  At the tightest radius (0.02) it is 6.1% against 0.9% (labelled) and 4.3% against 1.1% (detector). So matching is clearly better than chance (about 70% of merged pairs are likely the same lesion at 0.04), but
  few lesions are found in both photos, because most spots are only labelled or detected from one angle. The number of spots could not be checked against a true count, because no photo set labels the same lesion across angles.
- **How many duplicates were there?** Adding up the three photos counted **4.1%** more spots than there are on the face (1061 to 1018 spots). The count changes only a little; the larger benefits are the 3D map,
  the coverage report, a place for each spot, and a base for following the same spot from scan to scan.

## Limits
- The 3D shape is an **average face fitted to the landmarks**, not a scan: positions are approximate (millimetres to a few centimetres at the hairline and the far side of a turned face). Use it for "where on the face", not for measuring.
- Only the part of the face that some photo sees is real texture; the rest of the 3D map is a smooth skin-tone fill.
- Needs the whole face in view at less than about 65 degrees of head turn.
- The numbers above are from the platform's built-in spot detector; a better detector gives more matches.
