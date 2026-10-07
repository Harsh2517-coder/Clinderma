"""Stage 1 of the pipeline: illumination / local-contrast enhancement."""
import cv2
import numpy as np

import config as cfg


def apply_clahe(img_bgr, clip=None, tile=None):
    """CLAHE on the LAB L channel only (a/b untouched, so redness and skin tone aren't distorted), after a gated global
    exposure correction (see config.EXPOSURE_*)."""
    clip = cfg.CLAHE_CLIP_LIMIT if clip is None else clip
    tile = cfg.CLAHE_TILE_GRID_SIZE if tile is None else tile
    l, a, b = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB))
    med = float(np.median(l))
    if cfg.EXPOSURE_NORMALIZE and 1 < med < 254 and abs(med - cfg.EXPOSURE_TARGET_L) > cfg.EXPOSURE_TOLERANCE:
        g = np.clip(np.log(cfg.EXPOSURE_TARGET_L / 255) / np.log(med / 255), *cfg.EXPOSURE_GAMMA_RANGE)
        l = cv2.LUT(l, (255 * (np.arange(256) / 255) ** g).astype(np.uint8))
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
