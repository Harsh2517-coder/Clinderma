#!/usr/bin/env python3
"""Generate five deterministic Clinderma face-region masks for one image.

The script runs the same pretrained MediaPipe Face Mesh setup used by
detect_face_landmarks.py, then converts landmark coordinates into simple,
editable anatomical regions. It does not train a model and does not read or
modify ACNE04 annotations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


REGION_NAMES = ("forehead", "left_cheek", "right_cheek", "nose", "chin")


@dataclass(frozen=True)
class RegionColors:
    forehead: tuple[int, int, int] = (64, 180, 255)
    left_cheek: tuple[int, int, int] = (80, 220, 120)
    right_cheek: tuple[int, int, int] = (255, 170, 70)
    nose: tuple[int, int, int] = (230, 90, 220)
    chin: tuple[int, int, int] = (80, 140, 255)


@dataclass(frozen=True)
class RegionGeometry:
    """Editable landmark definitions for this first deterministic version.

    Landmark notes:
    - FACE_OVAL clips every region to the detected facial outline.
    - FOREHEAD uses the top of the face oval and stops near the eyebrow line.
      MediaPipe does not provide a true hairline, so landmark 10 and the upper
      oval are treated as the visible upper-forehead boundary.
    - NOSE uses bridge, tip, nostril, and sidewall landmarks. The convex hull
      is dilated and not clipped to FACE_OVAL because side-view noses can sit
      outside MediaPipe's oval contour.
    - CHIN starts below the lower lip and extends to landmark 152 at the chin.
    - CHEEKS are the remaining mid-face pixels inside FACE_OVAL after removing
      forehead, nose, and chin, split by the visible face center. Region names
      are anatomical: patient's left cheek is on image right.
    """

    forehead_boundary_landmarks: tuple[int, ...] = (70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300)
    nose_landmarks: tuple[int, ...] = (
        168,
        6,
        197,
        195,
        5,
        4,
        45,
        51,
        98,
        2,
        327,
        281,
        275,
        440,
        344,
        417,
        351,
        412,
        399,
        419,
        248,
        188,
        122,
        193,
    )
    lower_lip_landmarks: tuple[int, ...] = (17, 18, 200)
    chin_landmark: int = 152
    midface_top_landmarks: tuple[int, ...] = (70, 105, 107, 9, 336, 334, 300)
    chin_start_ratio: float = 0.38
    nose_dilation_ratio: float = 0.035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect a face in one image, generate five Clinderma anatomical "
            "region masks, and save mask PNGs plus an overlay visualization."
        )
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to one input image, for example examples/levle0_2.jpg.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for MediaPipe face detection.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Opacity for colored region overlays, from 0.0 to 1.0.",
    )
    return parser.parse_args()


def detect_face_landmarks(image_bgr: np.ndarray, min_detection_confidence: float):
    """Run the same MediaPipe Face Mesh configuration as detect_face_landmarks.py."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        raise RuntimeError(
            "No face landmarks detected. Try another examples image or lower "
            "--min-detection-confidence."
        )

    return results.multi_face_landmarks[0]


def landmarks_to_pixels(face_landmarks, width: int, height: int) -> np.ndarray:
    points = []
    for landmark in face_landmarks.landmark:
        x = int(round(landmark.x * (width - 1)))
        y = int(round(landmark.y * (height - 1)))
        points.append((np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)))
    return np.asarray(points, dtype=np.int32)


def fill_polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def face_oval_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mp_face_mesh = mp.solutions.face_mesh
    ordered_indices = ordered_connection_path(mp_face_mesh.FACEMESH_FACE_OVAL)
    polygon = points[ordered_indices]
    return fill_polygon_mask(shape, polygon)


def ordered_connection_path(connections: frozenset[tuple[int, int]]) -> list[int]:
    """Convert MediaPipe contour connections into an ordered landmark path."""
    neighbors: dict[int, list[int]] = {}
    for start, end in connections:
        neighbors.setdefault(start, []).append(end)
        neighbors.setdefault(end, []).append(start)

    start = min(neighbors)
    path = [start]
    previous = None
    current = start
    while True:
        candidates = [node for node in neighbors[current] if node != previous]
        if not candidates:
            break
        next_node = candidates[0]
        if next_node == start:
            break
        path.append(next_node)
        previous, current = current, next_node
        if len(path) > len(neighbors):
            break
    return path


def horizontal_band_mask(
    shape: tuple[int, int],
    face_mask: np.ndarray,
    y_min: int | None = None,
    y_max: int | None = None,
) -> np.ndarray:
    height, width = shape
    band = np.zeros(shape, dtype=np.uint8)
    top = 0 if y_min is None else int(np.clip(y_min, 0, height - 1))
    bottom = height - 1 if y_max is None else int(np.clip(y_max, 0, height - 1))
    if bottom >= top:
        band[top : bottom + 1, :width] = 255
    return cv2.bitwise_and(face_mask, band)


def generate_region_masks(
    points: np.ndarray,
    image_shape: tuple[int, int, int],
    geometry: RegionGeometry,
) -> dict[str, np.ndarray]:
    height, width = image_shape[:2]
    shape = (height, width)
    face_mask = face_oval_mask(shape, points)
    face_ys, face_xs = np.where(face_mask > 0)
    if len(face_xs) == 0:
        raise RuntimeError("Face oval mask is empty; cannot generate regions.")

    face_width = int(face_xs.max() - face_xs.min() + 1)
    face_center_x = int((face_xs.min() + face_xs.max()) / 2)

    brow_y = int(np.mean(points[list(geometry.forehead_boundary_landmarks), 1]))
    midface_top_y = int(np.mean(points[list(geometry.midface_top_landmarks), 1]))
    lower_lip_y = int(np.mean(points[list(geometry.lower_lip_landmarks), 1]))
    chin_y = int(points[geometry.chin_landmark, 1])
    chin_start_y = int(lower_lip_y + geometry.chin_start_ratio * (chin_y - lower_lip_y))

    forehead = horizontal_band_mask(shape, face_mask, y_max=brow_y)

    nose_hull = cv2.convexHull(points[list(geometry.nose_landmarks)])
    nose = fill_polygon_mask(shape, nose_hull[:, 0, :])
    dilation = max(5, int(round(face_width * geometry.nose_dilation_ratio)))
    if dilation % 2 == 0:
        dilation += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    nose = cv2.dilate(nose, kernel, iterations=1)

    chin = horizontal_band_mask(shape, face_mask, y_min=chin_start_y)

    midface = horizontal_band_mask(shape, face_mask, y_min=midface_top_y, y_max=chin_start_y)
    midface = cv2.bitwise_and(midface, cv2.bitwise_not(nose))
    midface = cv2.bitwise_and(midface, cv2.bitwise_not(forehead))
    midface = cv2.bitwise_and(midface, cv2.bitwise_not(chin))

    image_left = np.zeros(shape, dtype=np.uint8)
    image_right = np.zeros(shape, dtype=np.uint8)
    image_left[:, :face_center_x] = 255
    image_right[:, face_center_x:] = 255

    # Anatomical naming: patient's left cheek appears on the right side of the
    # image for standard face photographs; patient's right cheek appears left.
    left_cheek = cv2.bitwise_and(midface, image_right)
    right_cheek = cv2.bitwise_and(midface, image_left)

    return {
        "forehead": forehead,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
        "nose": nose,
        "chin": chin,
    }


def overlay_regions(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    alpha: float,
    colors: RegionColors,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = image_bgr.copy()

    for name in REGION_NAMES:
        mask = masks[name] > 0
        color = np.asarray(getattr(colors, name), dtype=np.float32)
        overlay[mask] = (
            (1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * color
        ).astype(np.uint8)

    for name in REGION_NAMES:
        contours, _ = cv2.findContours(masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, getattr(colors, name), 2)

    return overlay


def save_masks(masks: dict[str, np.ndarray], output_dir: Path, image_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in REGION_NAMES:
        path = output_dir / f"{image_stem}_{name}_mask.png"
        ok = cv2.imwrite(str(path), masks[name])
        if not ok:
            raise RuntimeError(f"OpenCV could not write mask: {path}")


def main() -> int:
    args = parse_args()
    image_path = args.image_path.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read the input image: {image_path}")

    face_landmarks = detect_face_landmarks(image_bgr, args.min_detection_confidence)
    height, width = image_bgr.shape[:2]
    points = landmarks_to_pixels(face_landmarks, width, height)
    masks = generate_region_masks(points, image_bgr.shape, RegionGeometry())
    overlay = overlay_regions(image_bgr, masks, args.alpha, RegionColors())

    output_root = Path(__file__).resolve().parent / "outputs"
    mask_dir = output_root / f"{image_path.stem}_region_masks"
    save_masks(masks, mask_dir, image_path.stem)

    overlay_path = output_root / f"{image_path.stem}_face_regions.jpg"
    output_root.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(overlay_path), overlay)
    if not ok:
        raise RuntimeError(f"OpenCV could not write overlay: {overlay_path}")

    print(f"Saved region overlay to: {overlay_path}")
    print(f"Saved five region masks to: {mask_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
