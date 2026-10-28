"""Builds the small logo files from the full logo. Put the full logo (monogram on top, 'FaceVeda' wordmark below) at static/logo.png, then run:
    python3 make_logo_assets.py
Creates static/logo-mark.png (the F/V monogram only, square, for the top bar) and static/favicon.png."""
import os, sys
import cv2, numpy as np

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"); src = os.path.join(S, "logo.png")
if not os.path.exists(src): sys.exit(f"Save the full logo as {src} first.")
im = cv2.imread(src, cv2.IMREAD_COLOR); bg = np.median(im[:10].reshape(-1, 3), axis=0)
ink = (np.abs(im.astype(int) - bg).sum(axis=2) > 60); rows = np.nonzero(ink.any(axis=1))[0]
# the monogram is the first band of ink rows; the wordmark starts after the first blank gap of at least 0.5% of the height
gap = max(4, im.shape[0] // 200); end = rows[0]
for a, b in zip(rows, rows[1:]):
    if b - a > gap: break
    end = b
else: end = rows[-1]
band = ink[rows[0]:end + 1]; cols = np.nonzero(band.any(axis=0))[0]; crop = im[rows[0]:end + 1, cols[0]:cols[-1] + 1]
side = int(max(crop.shape[:2]) * 1.12); canvas = np.full((side, side, 3), bg, np.uint8); y0, x0 = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
cv2.imwrite(os.path.join(S, "logo-mark.png"), cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_AREA)); cv2.imwrite(os.path.join(S, "favicon.png"), cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_AREA))
print("wrote logo-mark.png and favicon.png from", crop.shape[:2], "monogram area")
