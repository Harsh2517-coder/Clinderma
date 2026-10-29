#!/usr/bin/env python3
"""Generate Clinderma v2 facial-region masks for one ACNE04 image.

This implementation intentionally does not reuse the v1 polygon layout. It uses
two pretrained signals:

1. MediaPipe Face Mesh for face orientation, landmarks, and debug boundaries.
2. A pretrained face-parsing model for semantic skin/nose/hair/eye/brow/lip
   masks. The default model is jonathandinu/face-parsing, a SegFormer face
   parser trained on CelebAMask-HQ labels.

MediaPipe landmarks are not treated as semantic segmentation. They are used to
divide the parsed visible skin into forehead, cheeks, nose, and chin zones.
This is still an anatomical validation prototype, not a clinically validated
segmentation model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from PIL import Image


REGION_NAMES = ("forehead", "left_cheek", "right_cheek", "nose", "chin")
DEFAULT_MODEL_ID = "jonathandinu/face-parsing"


class View(str, Enum):
    FRONT = "FRONT"
    LEFT_PROFILE = "LEFT_PROFILE"
    RIGHT_PROFILE = "RIGHT_PROFILE"


@dataclass(frozen=True)
class RegionColors:
    forehead: tuple[int, int, int] = (40, 190, 255)
    left_cheek: tuple[int, int, int] = (80, 220, 110)
    right_cheek: tuple[int, int, int] = (255, 170, 70)
    nose: tuple[int, int, int] = (220, 70, 220)
    chin: tuple[int, int, int] = (70, 150, 255)


@dataclass(frozen=True)
class OrientationThresholds:
    """Thresholds for orientation classification.

    nose_shift_ratio = (nose-tip x - face-center x) / visible-face width.
    eye_width_ratio = min(projected eye width) / max(projected eye width).

    The nose shift determines direction; the eye-width ratio confirms whether
    one eye is substantially foreshortened. The ACNE04 examples include
    three-quarter views, so FRONT allows moderate nose shift when both eyes are
    still comparably visible.
    """

    front_eye_width_ratio_min: float = 0.68
    profile_eye_width_ratio_max: float = 0.62
    strong_profile_nose_shift_min: float = 0.52


@dataclass(frozen=True)
class BoundaryConfig:
    """Landmark boundaries for splitting parsed facial skin.

    These are not final anatomical truth. They are adjustable separators applied
    to pretrained face-parsing skin/nose masks:

    - forehead_lower_curve follows left brow -> glabella -> right brow. Skin
      above this curve, while excluding hair/eyes/brows, becomes forehead.
    - cheek_upper_curve follows the lower orbital boundary so cheeks do not
      include eyes or lower eyelids where landmarks permit.
    - chin_upper_curve follows the lower mouth/lip boundary; parsed skin below
      this curve down to the visible chin/jaw contour becomes chin.
    - nose_extra_landmarks cover bridge, tip, alae, and nostril-side landmarks.
      The parser's nose class is unioned with this landmark support to better
      cover side-profile noses.
    - facial_midline follows glabella -> nose -> philtrum -> lower lip -> chin.
      Frontal cheek masks are separated relative to this anatomical midline,
      not by a rectangular image split.
    """

    forehead_lower_curve: tuple[int, ...] = (127, 70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300, 356)
    cheek_upper_curve: tuple[int, ...] = (234, 116, 123, 147, 187, 207, 216, 212, 202, 43, 106, 182, 83, 18, 313, 406, 335, 273, 422, 432, 436, 427, 411, 376, 352, 345, 454)
    chin_upper_curve: tuple[int, ...] = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
    mouth_exclusion: tuple[int, ...] = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
    nose_extra_landmarks: tuple[int, ...] = (
        6, 168, 197, 195, 5, 4, 1, 2, 98, 97, 94, 327, 326, 328,
        45, 51, 115, 220, 275, 281, 344, 440, 193, 122, 188, 412, 351, 417
    )
    facial_midline: tuple[int, ...] = (10, 9, 168, 6, 197, 4, 2, 0, 17, 18, 200, 152)
    left_face_side: tuple[int, ...] = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152)
    right_face_side: tuple[int, ...] = (10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate v2 Clinderma masks for one image using MediaPipe "
            "orientation plus pretrained face parsing."
        )
    )
    parser.add_argument("image_path", type=Path, help="Path to one ACNE04 image.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face face-parsing model id.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def detect_face_landmarks(image_bgr: np.ndarray, min_detection_confidence: float):
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
        raise RuntimeError("No MediaPipe face landmarks detected.")
    return results.multi_face_landmarks[0]


def landmarks_to_pixels(face_landmarks, width: int, height: int) -> np.ndarray:
    points = []
    for landmark in face_landmarks.landmark:
        x = int(round(landmark.x * (width - 1)))
        y = int(round(landmark.y * (height - 1)))
        points.append((np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)))
    return np.asarray(points, dtype=np.int32)


def ordered_connection_path(connections: frozenset[tuple[int, int]]) -> list[int]:
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


def fill_polygon(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def face_oval_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    indices = ordered_connection_path(mp.solutions.face_mesh.FACEMESH_FACE_OVAL)
    return fill_polygon(shape, points[indices])


def polygon_from_curve_to_border(
    shape: tuple[int, int],
    curve_points: np.ndarray,
    side: str,
) -> np.ndarray:
    height, width = shape
    curve = curve_points.astype(np.int32)
    if side == "top":
        border = np.array([[width - 1, 0], [0, 0]], dtype=np.int32)
        return np.vstack([curve, border])
    if side == "bottom":
        border = np.array([[width - 1, height - 1], [0, height - 1]], dtype=np.int32)
        return np.vstack([curve, border])
    raise ValueError(f"Unknown side: {side}")


def semantic_face_parse(image_bgr: np.ndarray, model_id: str) -> tuple[np.ndarray, dict[int, str]]:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, SegformerImageProcessor
    except ImportError as exc:
        raise RuntimeError(
            "generate_face_regions_v2.py requires torch and transformers. "
            "Install clinderma_preprocessing/requirements.txt again."
        ) from exc

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
    except ValueError:
        # jonathandinu/face-parsing is a SegFormer checkpoint whose repository
        # does not currently expose the newer AutoImageProcessor metadata key.
        processor = SegformerImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
    model.eval()

    with torch.no_grad():
        inputs = processor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        labels = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    id2label = {int(key): value for key, value in model.config.id2label.items()}
    return labels, id2label


def labels_to_mask(labels: np.ndarray, id2label: dict[int, str], names: set[str]) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=np.uint8)
    normalized = {name.lower() for name in names}
    for label_id, label_name in id2label.items():
        if label_name.lower() in normalized:
            mask[labels == label_id] = 255
    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    num, components, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(components == largest, 255, 0).astype(np.uint8)


def clean_mask(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def split_mask_by_midline(
    shape: tuple[int, int],
    points: np.ndarray,
    midline_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return masks on image-left and image-right of a landmark midline curve."""
    height, width = shape
    midline = points[list(midline_indices)].astype(np.float32)
    midline = midline[np.argsort(midline[:, 1])]
    ys = np.arange(height, dtype=np.float32)
    line_x = np.interp(ys, midline[:, 1], midline[:, 0], left=midline[0, 0], right=midline[-1, 0])

    xx = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    boundary = np.tile(line_x[:, None], (1, width))
    image_left = np.where(xx < boundary, 255, 0).astype(np.uint8)
    image_right = np.where(xx >= boundary, 255, 0).astype(np.uint8)
    return image_left, image_right


def classify_view(points: np.ndarray, face_mask: np.ndarray, thresholds: OrientationThresholds) -> tuple[View, dict[str, float]]:
    ys, xs = np.where(face_mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Empty face mask; cannot classify view.")
    face_center_x = float((xs.min() + xs.max()) / 2.0)
    face_width = float(xs.max() - xs.min() + 1)
    nose_shift_ratio = float((points[1, 0] - face_center_x) / face_width)

    # MediaPipe common eye pairs: 33-133 and 362-263. The ratio is used as a
    # visibility/asymmetry signal rather than as an anatomy label.
    eye_a_width = float(np.linalg.norm(points[33] - points[133]))
    eye_b_width = float(np.linalg.norm(points[362] - points[263]))
    eye_width_ratio = min(eye_a_width, eye_b_width) / max(eye_a_width, eye_b_width, 1e-6)

    if eye_width_ratio >= thresholds.front_eye_width_ratio_min:
        view = View.FRONT
    elif abs(nose_shift_ratio) >= thresholds.strong_profile_nose_shift_min or eye_width_ratio <= thresholds.profile_eye_width_ratio_max:
        # Negative shift means the nose projects toward image-left; positive
        # shift means image-right. This names the view direction, not a cheek
        # mask by image side.
        view = View.LEFT_PROFILE if nose_shift_ratio < 0 else View.RIGHT_PROFILE
    else:
        view = View.FRONT

    return view, {
        "nose_shift_ratio": nose_shift_ratio,
        "eye_width_ratio": eye_width_ratio,
        "eye_a_width_px": eye_a_width,
        "eye_b_width_px": eye_b_width,
        "face_width_px": face_width,
    }


def build_region_masks(
    image_shape: tuple[int, int, int],
    points: np.ndarray,
    parsing_labels: np.ndarray,
    id2label: dict[int, str],
    config: BoundaryConfig,
    view: View,
) -> dict[str, np.ndarray]:
    shape = image_shape[:2]
    oval = face_oval_mask(shape, points)

    skin = labels_to_mask(parsing_labels, id2label, {"skin"})
    nose_semantic = labels_to_mask(parsing_labels, id2label, {"nose"})
    exclusions = labels_to_mask(
        parsing_labels,
        id2label,
        {"l_eye", "r_eye", "l_brow", "r_brow", "mouth", "u_lip", "l_lip", "hair", "hat", "eye_g"},
    )

    visible_skin = cv2.bitwise_and(clean_mask(skin, 7), oval)
    visible_skin = cv2.bitwise_and(visible_skin, cv2.bitwise_not(exclusions))
    visible_skin = largest_component(visible_skin)

    nose_hull = cv2.convexHull(points[list(config.nose_extra_landmarks)])
    nose_support = fill_polygon(shape, nose_hull[:, 0, :])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    nose_support = cv2.dilate(nose_support, kernel, iterations=1)
    nose = cv2.bitwise_or(nose_semantic, cv2.bitwise_and(nose_support, cv2.bitwise_or(skin, nose_semantic)))
    nose = clean_mask(nose, 7)

    forehead_curve = points[list(config.forehead_lower_curve)]
    forehead_limit = fill_polygon(shape, polygon_from_curve_to_border(shape, forehead_curve, "top"))
    forehead = cv2.bitwise_and(visible_skin, forehead_limit)

    chin_curve = points[list(config.chin_upper_curve)]
    chin_limit = fill_polygon(shape, polygon_from_curve_to_border(shape, chin_curve, "bottom"))
    chin = cv2.bitwise_and(visible_skin, chin_limit)

    mouth_poly = fill_polygon(shape, cv2.convexHull(points[list(config.mouth_exclusion)])[:, 0, :])
    mouth_poly = cv2.dilate(mouth_poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 25)), iterations=1)

    cheek_base = visible_skin.copy()
    for forbidden in (forehead, chin, nose, mouth_poly):
        cheek_base = cv2.bitwise_and(cheek_base, cv2.bitwise_not(forbidden))
    cheek_base = clean_mask(cheek_base, 9)

    left_cheek = np.zeros(shape, dtype=np.uint8)
    right_cheek = np.zeros(shape, dtype=np.uint8)

    image_left_of_midline, image_right_of_midline = split_mask_by_midline(
        shape, points, config.facial_midline
    )
    profile_cheek_support = cv2.bitwise_or(
        fill_polygon(shape, points[list(config.left_face_side)]),
        fill_polygon(shape, points[list(config.right_face_side)]),
    )

    if view == View.FRONT:
        # Anatomical naming for frontal photographs: the patient's left cheek
        # appears image-right, and the patient's right cheek appears image-left.
        left_cheek = cv2.bitwise_and(cheek_base, image_right_of_midline)
        right_cheek = cv2.bitwise_and(cheek_base, image_left_of_midline)
    elif view == View.LEFT_PROFILE:
        left_cheek = cv2.bitwise_and(cheek_base, profile_cheek_support)
    elif view == View.RIGHT_PROFILE:
        right_cheek = cv2.bitwise_and(cheek_base, profile_cheek_support)

    masks = {
        "forehead": clean_mask(forehead, 7),
        "left_cheek": largest_component(clean_mask(left_cheek, 7)),
        "right_cheek": largest_component(clean_mask(right_cheek, 7)),
        "nose": largest_component(nose),
        "chin": largest_component(clean_mask(cv2.bitwise_and(chin, cv2.bitwise_not(mouth_poly)), 7)),
    }

    for name in REGION_NAMES:
        masks[name] = np.where(masks[name] > 0, 255, 0).astype(np.uint8)
    return masks


def overlay_regions(image_bgr: np.ndarray, masks: dict[str, np.ndarray], alpha: float) -> np.ndarray:
    colors = RegionColors()
    overlay = image_bgr.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    for name in REGION_NAMES:
        mask = masks[name] > 0
        color = np.asarray(getattr(colors, name), dtype=np.float32)
        overlay[mask] = ((1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * color).astype(np.uint8)
    for name in REGION_NAMES:
        contours, _ = cv2.findContours(masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, getattr(colors, name), 3)
    return overlay


def draw_debug(
    image_bgr: np.ndarray,
    points: np.ndarray,
    masks: dict[str, np.ndarray],
    view: View,
    metrics: dict[str, float],
    config: BoundaryConfig,
) -> np.ndarray:
    debug = image_bgr.copy()
    mp_drawing = mp.solutions.drawing_utils
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    height, width = image_bgr.shape[:2]
    for point in points:
        face_landmarks_proto.landmark.add(x=float(point[0]) / width, y=float(point[1]) / height, z=0.0)

    mp_drawing.draw_landmarks(
        image=debug,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1),
    )
    cv2.polylines(debug, [points[list(config.forehead_lower_curve)]], False, (0, 255, 255), 4)
    cv2.polylines(debug, [points[list(config.chin_upper_curve)]], False, (255, 255, 0), 4)
    cv2.polylines(debug, [points[list(config.facial_midline)]], False, (255, 255, 255), 4)
    cv2.polylines(debug, [cv2.convexHull(points[list(config.nose_extra_landmarks)])[:, 0, :]], True, (255, 0, 255), 4)
    for name in REGION_NAMES:
        contours, _ = cv2.findContours(masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, contours, -1, getattr(RegionColors(), name), 2)

    text = (
        f"{view.value}  nose_shift={metrics['nose_shift_ratio']:.3f}  "
        f"eye_ratio={metrics['eye_width_ratio']:.3f}"
    )
    cv2.putText(debug, text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(debug, text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    return debug


def save_outputs(
    image_stem: str,
    output_root: Path,
    masks: dict[str, np.ndarray],
    overlay: np.ndarray,
    debug: np.ndarray,
    view: View,
    metrics: dict[str, float],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    mask_dir = output_root / f"{image_stem}_v2_region_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    pixel_counts: dict[str, int] = {}
    for name in REGION_NAMES:
        mask_path = mask_dir / f"{image_stem}_{name}_mask.png"
        cv2.imwrite(str(mask_path), masks[name])
        pixel_counts[name] = int(cv2.countNonZero(masks[name]))

    overlay_path = output_root / f"{image_stem}_v2_face_regions.jpg"
    debug_path = output_root / f"{image_stem}_v2_debug_boundaries.jpg"
    report_path = output_root / f"{image_stem}_v2_view_report.json"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(debug_path), debug)

    report = {
        "image": image_stem,
        "view": view.value,
        "orientation_metrics": metrics,
        "mask_pixel_counts": pixel_counts,
        "outputs": {
            "overlay": str(overlay_path),
            "debug": str(debug_path),
            "masks": str(mask_dir),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    image_path = args.image_path.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    landmarks = detect_face_landmarks(image_bgr, args.min_detection_confidence)
    height, width = image_bgr.shape[:2]
    points = landmarks_to_pixels(landmarks, width, height)
    oval = face_oval_mask((height, width), points)
    view, metrics = classify_view(points, oval, OrientationThresholds())

    parsing_labels, id2label = semantic_face_parse(image_bgr, args.model_id)
    config = BoundaryConfig()
    masks = build_region_masks(image_bgr.shape, points, parsing_labels, id2label, config, view)
    overlay = overlay_regions(image_bgr, masks, args.alpha)
    debug = draw_debug(image_bgr, points, masks, view, metrics, config)

    report = save_outputs(
        image_stem=image_path.stem,
        output_root=Path(__file__).resolve().parent / "outputs",
        masks=masks,
        overlay=overlay,
        debug=debug,
        view=view,
        metrics=metrics,
    )

    print(json.dumps({"view": report["view"], "mask_pixel_counts": report["mask_pixel_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
