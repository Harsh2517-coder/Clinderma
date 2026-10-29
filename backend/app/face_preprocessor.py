import os
import cv2
import numpy as np
import base64
import io
import logging
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
from PIL import Image

logger = logging.getLogger("clinderma.face_preprocessor")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    mp = None
    MP_AVAILABLE = False

REGION_NAMES = ("forehead", "left_cheek", "right_cheek", "nose", "chin")

# Predefined canonical counter-clockwise indices of the MediaPipe Face Mesh oval
FACEMESH_FACE_OVAL_INDICES = [
    10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149,
    176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389,
    251, 284, 332, 297, 338
]


class View(str, Enum):
    FRONT = "FRONT"
    LEFT_PROFILE = "LEFT_PROFILE"
    RIGHT_PROFILE = "RIGHT_PROFILE"


@dataclass(frozen=True)
class RegionColors:
    forehead: tuple[int, int, int] = (40, 190, 255)     # Amber/Gold in BGR (255, 190, 40)
    left_cheek: tuple[int, int, int] = (80, 220, 110)   # Green in BGR (110, 220, 80)
    right_cheek: tuple[int, int, int] = (255, 170, 70)  # Blue/Orange in BGR (70, 170, 255)
    nose: tuple[int, int, int] = (220, 70, 220)         # Magenta in BGR (220, 70, 220)
    chin: tuple[int, int, int] = (70, 150, 255)         # Sky Blue in BGR (255, 150, 70)


@dataclass
class NormalizedLandmarkPoint:
    x: float
    y: float
    z: float = 0.0


class NormalizedLandmarksList:
    def __init__(self, landmarks: List[Any]):
        self.landmark = landmarks


class FacePreprocessor:
    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self.colors = RegionColors()
        self._landmarker = None
        self._init_detector()

    def _init_detector(self):
        """Initializes MediaPipe detector supporting both modern Tasks Vision API and legacy Solutions API."""
        if not MP_AVAILABLE or mp is None:
            logger.info("MediaPipe module not found; running with robust fallback geometry.")
            return

        # Attempt modern Tasks API (mediapipe >= 0.10.30 / 1.x)
        if hasattr(mp, "tasks") and hasattr(mp.tasks, "python"):
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                candidate_paths = [
                    os.path.join(cur_dir, "..", "..", "preprocessing", "Clinderma", "V_3", "models", "face_landmarker.task"),
                    os.path.join(cur_dir, "models", "face_landmarker.task"),
                    os.path.join(cur_dir, "face_landmarker.task"),
                    "face_landmarker.task",
                ]
                model_file = None
                for p in candidate_paths:
                    if os.path.exists(p):
                        model_file = os.path.abspath(p)
                        break

                if model_file:
                    base_options = mp_python.BaseOptions(model_asset_path=model_file)
                    options = mp_vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        num_faces=1,
                        min_face_detection_confidence=self.min_detection_confidence,
                    )
                    self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                    logger.info(f"Loaded MediaPipe Tasks FaceLandmarker model from {model_file}")
            except Exception as e:
                logger.warning(f"Could not load MediaPipe FaceLandmarker Tasks API: {e}")

    def _fallback_detect_landmarks(self, image_bgr: np.ndarray) -> Optional[NormalizedLandmarksList]:
        """Synthesize landmark points from face detector bounding box when MediaPipe fails or is unavailable."""
        h, w = image_bgr.shape[:2]
        box = None

        try:
            from face_crop import detect_faces
            boxes = detect_faces(image_bgr, conf=0.25)
            if boxes:
                box = boxes[0][:4]
        except Exception:
            pass

        if box is None:
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    box = (float(fx), float(fy), float(fx + fw), float(fy + fh))
            except Exception:
                pass

        if box is None:
            # Fallback assuming central area of image
            box = (float(w * 0.05), float(h * 0.05), float(w * 0.95), float(h * 0.95))

        x0, y0, x1, y1 = box
        bw = max(10.0, x1 - x0)
        bh = max(10.0, y1 - y0)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        # Construct 478 synthetic landmark points mapped to normalized coordinates
        synthetic = [NormalizedLandmarkPoint(cx / w, cy / h) for _ in range(478)]

        # Map oval landmarks to ellipse perimeter
        n_oval = len(FACEMESH_FACE_OVAL_INDICES)
        for i, idx in enumerate(FACEMESH_FACE_OVAL_INDICES):
            theta = 2.0 * np.pi * (i / n_oval)
            px = cx + (bw * 0.48) * np.sin(theta)
            py = cy + (bh * 0.50) * np.cos(theta)
            synthetic[idx] = NormalizedLandmarkPoint(
                float(np.clip(px / w, 0.0, 1.0)),
                float(np.clip(py / h, 0.0, 1.0)),
            )

        # Key anatomical landmark anchors
        synthetic[1] = NormalizedLandmarkPoint(cx / w, cy / h)
        synthetic[168] = NormalizedLandmarkPoint(cx / w, (cy - bh * 0.15) / h)
        synthetic[6] = NormalizedLandmarkPoint(cx / w, (cy - bh * 0.10) / h)
        synthetic[197] = NormalizedLandmarkPoint(cx / w, (cy - bh * 0.05) / h)
        synthetic[4] = NormalizedLandmarkPoint(cx / w, (cy + bh * 0.05) / h)
        synthetic[2] = NormalizedLandmarkPoint(cx / w, (cy + bh * 0.08) / h)
        synthetic[0] = NormalizedLandmarkPoint(cx / w, (cy + bh * 0.12) / h)

        eye_y = (cy - bh * 0.15) / h
        synthetic[33] = NormalizedLandmarkPoint((cx - bw * 0.30) / w, eye_y)
        synthetic[133] = NormalizedLandmarkPoint((cx - bw * 0.12) / w, eye_y)
        synthetic[362] = NormalizedLandmarkPoint((cx + bw * 0.12) / w, eye_y)
        synthetic[263] = NormalizedLandmarkPoint((cx + bw * 0.30) / w, eye_y)

        brow_y = (cy - bh * 0.22) / h
        for i, idx in enumerate((70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300)):
            fraction = -0.35 + 0.70 * (i / 10.0)
            synthetic[idx] = NormalizedLandmarkPoint((cx + bw * fraction) / w, brow_y)

        mouth_y = (cy + bh * 0.25) / h
        synthetic[17] = NormalizedLandmarkPoint(cx / w, mouth_y)
        synthetic[18] = NormalizedLandmarkPoint(cx / w, mouth_y + 0.02 * bh / h)
        synthetic[200] = NormalizedLandmarkPoint(cx / w, mouth_y + 0.04 * bh / h)
        synthetic[152] = NormalizedLandmarkPoint(cx / w, y1 / h)

        return NormalizedLandmarksList(synthetic)

    def detect_face_landmarks(self, image_bgr: np.ndarray):
        """Extract MediaPipe FaceMesh landmarks using Tasks API, legacy Solutions API, or fallback geometry."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 1. Try modern Tasks API (MediaPipe >= 0.10.30 / 1.x)
        if self._landmarker is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                results = self._landmarker.detect(mp_image)
                if results.face_landmarks and len(results.face_landmarks) > 0:
                    return NormalizedLandmarksList(results.face_landmarks[0])
            except Exception as e:
                logger.warning(f"MediaPipe FaceLandmarker error: {e}")

        # 2. Try legacy Solutions API (MediaPipe <= 0.10.14)
        if MP_AVAILABLE and mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                mp_face_mesh = mp.solutions.face_mesh
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self.min_detection_confidence,
                ) as face_mesh:
                    results = face_mesh.process(image_rgb)
                    if results.multi_face_landmarks:
                        return results.multi_face_landmarks[0]
            except Exception as e:
                logger.warning(f"MediaPipe FaceMesh Solutions error: {e}")

        # 3. Robust fallback landmark approximation
        return self._fallback_detect_landmarks(image_bgr)

    def landmarks_to_pixels(self, face_landmarks, width: int, height: int) -> np.ndarray:
        points = []
        lms = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
        for landmark in lms:
            x = int(round(landmark.x * (width - 1)))
            y = int(round(landmark.y * (height - 1)))
            points.append((np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)))
        return np.asarray(points, dtype=np.int32)

    def ordered_connection_path(self, connections=None) -> list[int]:
        if connections is None:
            return FACEMESH_FACE_OVAL_INDICES
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

    def fill_polygon(self, shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        return mask

    def face_oval_mask(self, shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
        indices = FACEMESH_FACE_OVAL_INDICES
        return self.fill_polygon(shape, points[indices])

    def classify_view(self, points: np.ndarray, face_mask: np.ndarray) -> Tuple[View, Dict[str, float]]:
        ys, xs = np.where(face_mask > 0)
        if len(xs) == 0:
            return View.FRONT, {"nose_shift_ratio": 0.0, "eye_width_ratio": 1.0}

        face_center_x = float((xs.min() + xs.max()) / 2.0)
        face_width = float(xs.max() - xs.min() + 1)
        nose_shift_ratio = float((points[1, 0] - face_center_x) / max(face_width, 1.0))

        # MediaPipe common eye pairs: 33-133 and 362-263
        eye_a_width = float(np.linalg.norm(points[33] - points[133]))
        eye_b_width = float(np.linalg.norm(points[362] - points[263]))
        eye_width_ratio = min(eye_a_width, eye_b_width) / max(eye_a_width, eye_b_width, 1e-6)

        if eye_width_ratio >= 0.68:
            view = View.FRONT
        elif abs(nose_shift_ratio) >= 0.52 or eye_width_ratio <= 0.62:
            view = View.LEFT_PROFILE if nose_shift_ratio < 0 else View.RIGHT_PROFILE
        else:
            view = View.FRONT

        return view, {
            "nose_shift_ratio": round(nose_shift_ratio, 4),
            "eye_width_ratio": round(eye_width_ratio, 4),
            "eye_a_width_px": round(eye_a_width, 1),
            "eye_b_width_px": round(eye_b_width, 1),
            "face_width_px": round(face_width, 1),
        }

    def generate_regions(self, image_bgr: np.ndarray, points: np.ndarray, view: View) -> Dict[str, np.ndarray]:
        """Generate 5 Clinderma anatomical masks: forehead, left_cheek, right_cheek, nose, chin."""
        height, width = image_bgr.shape[:2]
        shape = (height, width)
        oval = self.face_oval_mask(shape, points)

        # 1. Nose Mask
        nose_landmarks = (
            168, 6, 197, 195, 5, 4, 45, 51, 98, 2, 327, 281, 275, 440,
            344, 417, 351, 412, 399, 419, 248, 188, 122, 193, 1, 94, 97
        )
        nose_pts = points[list(nose_landmarks)]
        nose_hull = cv2.convexHull(nose_pts)
        nose_mask = self.fill_polygon(shape, nose_hull[:, 0, :])
        k_nose = max(5, int(width * 0.035)) | 1
        nose_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_nose, k_nose))
        nose_mask = cv2.dilate(nose_mask, nose_kernel, iterations=1)

        # 2. Forehead Mask
        forehead_boundary_idx = (70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300)
        forehead_line = points[list(forehead_boundary_idx)]
        forehead_poly = np.vstack([
            forehead_line,
            [[width - 1, 0], [0, 0]]
        ])
        forehead_full = self.fill_polygon(shape, forehead_poly)
        forehead_mask = cv2.bitwise_and(forehead_full, oval)
        forehead_mask = cv2.bitwise_and(forehead_mask, cv2.bitwise_not(nose_mask))

        # 3. Chin Mask
        lower_lip_idx = (17, 18, 200)
        lip_y = float(np.mean([points[i][1] for i in lower_lip_idx]))
        chin_y = float(points[152][1])
        chin_start = int(lip_y + 0.35 * (chin_y - lip_y))

        chin_rect = np.zeros(shape, dtype=np.uint8)
        chin_rect[chin_start:, :] = 255
        chin_mask = cv2.bitwise_and(chin_rect, oval)
        chin_mask = cv2.bitwise_and(chin_mask, cv2.bitwise_not(nose_mask))

        # 4. Cheeks (Remaining visible face inside oval)
        cheeks_base = oval.copy()
        for forbidden in (forehead_mask, chin_mask, nose_mask):
            cheeks_base = cv2.bitwise_and(cheeks_base, cv2.bitwise_not(forbidden))

        midline_idx = (10, 9, 168, 6, 197, 4, 2, 0, 17, 18, 200, 152)
        midline = points[list(midline_idx)].astype(np.float32)
        midline = midline[np.argsort(midline[:, 1])]
        ys = np.arange(height, dtype=np.float32)
        line_x = np.interp(ys, midline[:, 1], midline[:, 0], left=midline[0, 0], right=midline[-1, 0])
        xx = np.tile(np.arange(width, dtype=np.float32), (height, 1))
        boundary = np.tile(line_x[:, None], (1, width))

        image_left = np.where(xx < boundary, 255, 0).astype(np.uint8)
        image_right = np.where(xx >= boundary, 255, 0).astype(np.uint8)

        if view == View.FRONT:
            left_cheek_mask = cv2.bitwise_and(cheeks_base, image_right)
            right_cheek_mask = cv2.bitwise_and(cheeks_base, image_left)
        elif view == View.LEFT_PROFILE:
            left_cheek_mask = cheeks_base
            right_cheek_mask = np.zeros(shape, dtype=np.uint8)
        else:  # RIGHT_PROFILE
            right_cheek_mask = cheeks_base
            left_cheek_mask = np.zeros(shape, dtype=np.uint8)

        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        masks = {
            "forehead": cv2.morphologyEx(forehead_mask, cv2.MORPH_OPEN, kernel_clean),
            "left_cheek": cv2.morphologyEx(left_cheek_mask, cv2.MORPH_OPEN, kernel_clean),
            "right_cheek": cv2.morphologyEx(right_cheek_mask, cv2.MORPH_OPEN, kernel_clean),
            "nose": cv2.morphologyEx(nose_mask, cv2.MORPH_OPEN, kernel_clean),
            "chin": cv2.morphologyEx(chin_mask, cv2.MORPH_OPEN, kernel_clean),
        }

        for name in REGION_NAMES:
            masks[name] = np.where(masks[name] > 0, 255, 0).astype(np.uint8)

        return masks

    def create_region_overlay(self, image_bgr: np.ndarray, masks: Dict[str, np.ndarray], alpha: float = 0.40) -> np.ndarray:
        overlay = image_bgr.copy()
        alpha = float(np.clip(alpha, 0.0, 1.0))
        for name in REGION_NAMES:
            mask = masks[name] > 0
            color = np.asarray(getattr(self.colors, name), dtype=np.float32)
            overlay[mask] = ((1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * color).astype(np.uint8)

        for name in REGION_NAMES:
            contours, _ = cv2.findContours(masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, getattr(self.colors, name), 2)

        return overlay

    def create_debug_visualization(
        self,
        image_bgr: np.ndarray,
        points: np.ndarray,
        masks: Dict[str, np.ndarray],
        view: View,
        metrics: Dict[str, float]
    ) -> np.ndarray:
        debug = image_bgr.copy()
        height, width = image_bgr.shape[:2]

        # Draw face mesh oval outline
        oval_indices = FACEMESH_FACE_OVAL_INDICES
        for i in range(len(oval_indices)):
            p1 = tuple(points[oval_indices[i]])
            p2 = tuple(points[oval_indices[(i + 1) % len(oval_indices)]])
            cv2.line(debug, p1, p2, (0, 255, 255), 2)

        # Draw midline
        midline_idx = (10, 9, 168, 6, 197, 4, 2, 0, 17, 18, 200, 152)
        for i in range(len(midline_idx) - 1):
            p1 = tuple(points[midline_idx[i]])
            p2 = tuple(points[midline_idx[i + 1]])
            cv2.line(debug, p1, p2, (0, 200, 255), 1)

        # Draw landmark points
        step = max(1, len(points) // 200)
        for pt in points[::step]:
            cv2.circle(debug, (int(pt[0]), int(pt[1])), 1, (220, 220, 220), -1)

        # Region boundaries
        for name in REGION_NAMES:
            contours, _ = cv2.findContours(masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug, contours, -1, getattr(self.colors, name), 2)

        # Orientation badge overlay
        text = f"View: {view.value} | Shift: {metrics.get('nose_shift_ratio', 0):.2f} | Eye Ratio: {metrics.get('eye_width_ratio', 0):.2f}"
        cv2.rectangle(debug, (20, 20), (min(width - 20, 680), 70), (20, 20, 20), -1)
        cv2.putText(debug, text, (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2, cv2.LINE_AA)
        return debug

    def assign_zone_for_bbox(self, bbox: Dict[str, float], masks: Dict[str, np.ndarray]) -> Tuple[str, Dict[str, float]]:
        """Map bounding box to best Clinderma facial zone by intersection and center coordinate."""
        x1 = max(0, int(round(bbox["x1"])))
        y1 = max(0, int(round(bbox["y1"])))
        x2 = int(round(bbox["x2"]))
        y2 = int(round(bbox["y2"]))

        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))

        center_zone = None
        for name in REGION_NAMES:
            mask = masks[name]
            h, w = mask.shape
            if 0 <= cy < h and 0 <= cx < w and mask[cy, cx] > 0:
                center_zone = name
                break

        overlaps = {}
        for name in REGION_NAMES:
            mask = masks[name]
            h, w = mask.shape
            sub_mask = mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            overlaps[name] = int(cv2.countNonZero(sub_mask)) if sub_mask.size > 0 else 0

        best_overlap_zone = max(overlaps, key=overlaps.get)
        box_area = max(1, (x2 - x1) * (y2 - y1))
        max_overlap_px = overlaps[best_overlap_zone]
        overlap_fraction = max_overlap_px / float(box_area)

        if center_zone:
            assigned_zone = center_zone
        elif overlap_fraction >= 0.15:
            assigned_zone = best_overlap_zone
        else:
            assigned_zone = "unassigned"

        return assigned_zone, overlaps

    def process(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs complete facial preprocessing and returns masks, overlays, metrics, and base64 images."""
        height, width = image_bgr.shape[:2]
        total_pixels = height * width

        landmarks = self.detect_face_landmarks(image_bgr)
        if landmarks is None:
            raise ValueError("No face detected in the image. Please upload a clear facial photograph.")

        points = self.landmarks_to_pixels(landmarks, width, height)
        oval = self.face_oval_mask((height, width), points)
        view, metrics = self.classify_view(points, oval)
        masks = self.generate_regions(image_bgr, points, view)

        overlay_bgr = self.create_region_overlay(image_bgr, masks)
        debug_bgr = self.create_debug_visualization(image_bgr, points, masks, view, metrics)

        pixel_counts = {}
        pixel_percentages = {}
        for name in REGION_NAMES:
            cnt = int(cv2.countNonZero(masks[name]))
            pixel_counts[name] = cnt
            pixel_percentages[name] = round((cnt / total_pixels) * 100, 2)

        _, overlay_buf = cv2.imencode(".jpg", overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(overlay_buf).decode("utf-8")

        _, debug_buf = cv2.imencode(".jpg", debug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        debug_b64 = "data:image/jpeg;base64," + base64.b64encode(debug_buf).decode("utf-8")

        face_mask = oval.copy()
        for m in masks.values():
            face_mask = cv2.bitwise_or(face_mask, m)

        ys, xs = np.where(face_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            face_bbox = {
                "x1": int(xs.min()),
                "y1": int(ys.min()),
                "x2": int(xs.max()),
                "y2": int(ys.max()),
            }
        else:
            face_bbox = {"x1": 0, "y1": 0, "x2": width, "y2": height}

        mask_data_urls = {}
        for name in REGION_NAMES:
            _, mask_buf = cv2.imencode(".png", masks[name])
            mask_data_urls[name] = "data:image/png;base64," + base64.b64encode(mask_buf).decode("utf-8")

        return {
            "view": view.value,
            "orientation_metrics": metrics,
            "pixel_counts": pixel_counts,
            "pixel_percentages": pixel_percentages,
            "masks": masks,
            "face_mask": face_mask,
            "face_bbox": face_bbox,
            "visualizations": {
                "overlay_data_url": overlay_b64,
                "debug_data_url": debug_b64,
                "mask_data_urls": mask_data_urls,
            }
        }
