import cv2
import numpy as np
import base64
from typing import Dict, Any, List, Optional
import uuid
import logging

from app.v3_preprocessor import V3Preprocessor, bgr_to_base64_jpeg
from app.face_preprocessor import FacePreprocessor, REGION_NAMES
from app.acne_detector import AcneDetector

logger = logging.getLogger("clinderma.pipeline")

# GAGS Zone Factors
ZONE_FACTORS = {
    "forehead": 2,
    "right_cheek": 2,
    "left_cheek": 2,
    "nose": 1,
    "chin": 1,
    "unassigned": 0,
}

# Lesion colors for direct OpenCV annotation (BGR)
LESION_COLORS_BGR = {
    "pustule": (0, 0, 255),      # Red
    "papule": (0, 165, 255),     # Orange
    "nodule": (180, 0, 180),     # Purple
    "cyst": (180, 0, 180),       # Purple
    "comedone": (0, 220, 240),   # Yellow/Gold
    "default": (255, 100, 0),    # Blue
}

def get_lesion_grade(lesion_class: str) -> int:
    lc = lesion_class.lower()
    if any(k in lc for k in ("comedone", "blackhead", "whitehead")):
        return 1
    elif "papule" in lc:
        return 2
    elif "pustule" in lc:
        return 3
    elif any(k in lc for k in ("nodule", "cyst")):
        return 4
    elif "acne" in lc:
        return 2
    return 1

def get_gags_severity(total_score: int) -> tuple[str, str]:
    if total_score == 0:
        return "Clear", "Clear skin. No clinically significant acne lesions detected."
    elif 1 <= total_score <= 18:
        return "Mild", "Mild acne presentation (GAGS 1–18). Typically manageable with topical retinoids, benzoyl peroxide, or gentle salicylic acid regimens."
    elif 19 <= total_score <= 30:
        return "Moderate", "Moderate acne presentation (GAGS 19–30). Commonly indicated for combination topical therapy and systemic antimicrobials or oral therapy under dermatological supervision."
    elif 31 <= total_score <= 38:
        return "Severe", "Severe acne presentation (GAGS 31–38). Dermatological consultation recommended for oral isotretinoin or targeted systemic therapies."
    else:
        return "Very Severe", "Very severe presentation (GAGS ≥39). Note: In a 5-facial-zone layout, maximum score is 32 (Severe); scores ≥39 occur when chest and back zones are included."


class ClindermaPipeline:
    def __init__(self):
        self.v3_preprocessor = V3Preprocessor()
        self.face_preprocessor = FacePreprocessor()
        self.detector = AcneDetector()
        logger.info("ClindermaPipeline initialized with mandatory V3 Preprocessing Pipeline.")

    def create_annotated_image(self, image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> str:
        """Draws exact bounding boxes and labels directly onto the image pixels with OpenCV."""
        annotated = image_bgr.copy()
        h, w = annotated.shape[:2]
        thickness = max(2, int(min(h, w) * 0.0025))
        font_scale = max(0.4, min(h, w) * 0.00035)

        for det in detections:
            bbox = det["bbox"]
            x1 = max(0, min(w - 1, int(round(bbox["x1"]))))
            y1 = max(0, min(h - 1, int(round(bbox["y1"]))))
            x2 = max(0, min(w, int(round(bbox["x2"]))))
            y2 = max(0, min(h, int(round(bbox["y2"]))))

            cls_name = det["class"]
            conf = det["confidence"]
            zone = det.get("zone", "")

            # Get color
            if zone == "unassigned":
                color = (130, 130, 130)  # Gray for non-facial / unassigned detections
            else:
                color = LESION_COLORS_BGR.get("default")
                for k, c in LESION_COLORS_BGR.items():
                    if k in cls_name.lower():
                        color = c
                        break

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label banner
            prefix = "[Outside] " if zone == "unassigned" else ""
            label = f"{prefix}{cls_name.title()} {int(round(conf*100))}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_y1 = max(0, y1 - th - 6)
            label_y2 = y1
            label_x2 = min(w, x1 + tw + 8)

            cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0) if zone != "unassigned" else (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    def run(
        self,
        image_bytes: bytes,
        confidence: Optional[float] = None,
        filter_non_facial: bool = True,
        iou_threshold: float = 0.30,
    ) -> Dict[str, Any]:
        """Runs the entire Clinderma pipeline with MANDATORY V3 Preprocessing:
        Sequence:
          Raw Image ->
          V3 Preprocessing Pipeline (End-to-End: CLAHE, YOLOv8 face detection, BiSeNet parsing, skin masking, candidate detection, cutout) ->
          Preprocessed Face Cutout Image ->
          Trained Acne Detection Model (Roboflow Hosted API / RF-DETR) ->
          Bounding Box Coordinate Mapping (Cutout Space -> Input Image Space) ->
          Facial Zone Association & GAGS Clinical Scoring ->
          Direct Pixel Annotation & Diagnostics
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image file: corrupted or invalid image format.")

        img_h, img_w = image_bgr.shape[:2]

        # 1. MANDATORY V3 Preprocessing Pipeline (Full end-to-end execution)
        # Note: If no face is detected or preprocessing fails, process() raises ValueError.
        # It NEVER falls back to sending the raw image to the model!
        v3_result = self.v3_preprocessor.process(image_bgr)

        cutout_image = v3_result["cutout_image"]
        cutout_bytes = v3_result["cutout_bytes"]
        ox, oy = v3_result["cutout_origin"]
        cut_h, cut_w = cutout_image.shape[:2]

        logger.info(
            f"Mandatory V3 Preprocessing succeeded. Delivering preprocessed face cutout "
            f"[{cut_w}x{cut_h}, {cutout_image.dtype}, origin=({ox}, {oy})] to trained acne detection model."
        )

        # 2. Acne Detection on the V3 Preprocessed Face Cutout ONLY
        # Model strictly receives cutout_bytes produced by V3 preprocessing.
        detection_result = self.detector.predict(
            cutout_bytes,
            confidence=confidence,
            iou_threshold=iou_threshold,
        )
        crop_detections = detection_result.get("detections", [])

        # 3. Map Bounding Box Coordinates from Cutout Space to Original Image Coordinates
        # (Exact implementation of V3 preprocess.to_input_coords)
        raw_detections = []
        for det in crop_detections:
            cbox = det["bbox"]
            mapped_x1 = max(0.0, min(float(img_w), round(cbox["x1"] + ox, 2)))
            mapped_y1 = max(0.0, min(float(img_h), round(cbox["y1"] + oy, 2)))
            mapped_x2 = max(0.0, min(float(img_w), round(cbox["x2"] + ox, 2)))
            mapped_y2 = max(0.0, min(float(img_h), round(cbox["y2"] + oy, 2)))

            raw_detections.append({
                "class": det["class"],
                "confidence": det["confidence"],
                "bbox": {
                    "x1": mapped_x1,
                    "y1": mapped_y1,
                    "x2": mapped_x2,
                    "y2": mapped_y2,
                }
            })

        # 4. Facial Zone Segmentation & Landmark Alignment for Clinical GAGS Scoring
        zone_preproc = self.face_preprocessor.process(image_bgr)
        masks = zone_preproc["masks"]
        view = zone_preproc["view"]
        orientation_metrics = zone_preproc["orientation_metrics"]
        pixel_counts = zone_preproc["pixel_counts"]
        pixel_percentages = zone_preproc["pixel_percentages"]
        visualizations = zone_preproc["visualizations"]
        face_mask = zone_preproc["face_mask"]

        # 5. Associate Detections with Facial Zones
        enriched_detections: List[Dict[str, Any]] = []
        excluded_non_facial: List[Dict[str, Any]] = []
        zone_counts = {z: 0 for z in REGION_NAMES}
        zone_counts["unassigned"] = 0
        zone_max_grades = {z: 0 for z in REGION_NAMES}
        zone_max_grades["unassigned"] = 0
        lesions_summary: Dict[str, int] = {}

        for idx, det in enumerate(raw_detections):
            bbox = det["bbox"]
            lesion_class = det["class"]
            conf = det["confidence"]

            assigned_zone, overlaps = self.face_preprocessor.assign_zone_for_bbox(bbox, masks)

            cx = int(round((bbox["x1"] + bbox["x2"]) / 2.0))
            cy = int(round((bbox["y1"] + bbox["y2"]) / 2.0))
            is_on_face = 0 <= cy < img_h and 0 <= cx < img_w and face_mask[cy, cx] > 0

            if is_on_face and assigned_zone == "unassigned":
                if view == "LEFT_PROFILE":
                    assigned_zone = "left_cheek"
                elif view == "RIGHT_PROFILE":
                    assigned_zone = "right_cheek"
                else:
                    assigned_zone = "left_cheek" if cx >= (img_w / 2.0) else "right_cheek"
            elif view == "LEFT_PROFILE" and assigned_zone == "right_cheek":
                assigned_zone = "left_cheek"
            elif view == "RIGHT_PROFILE" and assigned_zone == "left_cheek":
                assigned_zone = "right_cheek"

            grade = get_lesion_grade(lesion_class)

            det_obj = {
                "id": f"det_{idx + 1}_{uuid.uuid4().hex[:6]}",
                "class": lesion_class,
                "confidence": round(conf, 3),
                "bbox": bbox,
                "zone": assigned_zone,
                "gags_grade": grade,
                "zone_factor": ZONE_FACTORS.get(assigned_zone, 1),
                "overlaps": overlaps,
            }

            if assigned_zone == "unassigned" and filter_non_facial:
                excluded_non_facial.append(det_obj)
            else:
                zone_counts[assigned_zone] = zone_counts.get(assigned_zone, 0) + 1
                if grade > zone_max_grades.get(assigned_zone, 0):
                    zone_max_grades[assigned_zone] = grade

                lesions_summary[lesion_class] = lesions_summary.get(lesion_class, 0) + 1
                enriched_detections.append(det_obj)

        # 6. Generate Direct Pixel-Level Annotated Image
        annotated_data_url = self.create_annotated_image(image_bgr, enriched_detections)
        visualizations["annotated_detections_data_url"] = annotated_data_url

        # 7. Add V3 Preprocessing Stage Visualizations to Response
        stages_bgr = v3_result.get("stages_bgr", {})
        v3_stage_urls = {
            "1_original": bgr_to_base64_jpeg(stages_bgr.get("1_original", image_bgr)),
            "2_clahe": bgr_to_base64_jpeg(stages_bgr.get("2_clahe", image_bgr)),
            "3_face_skin_mask": bgr_to_base64_jpeg(stages_bgr.get("3_face_skin_mask", image_bgr)),
            "4_candidate_mask": bgr_to_base64_jpeg(stages_bgr.get("4_candidate_mask", image_bgr)),
            "5_final_boxes": bgr_to_base64_jpeg(stages_bgr.get("5_final_boxes", image_bgr)),
            "6_model_input_cutout": bgr_to_base64_jpeg(stages_bgr.get("6_model_input_cutout", cutout_image)),
        }
        visualizations["v3_stages"] = v3_stage_urls

        # 8. Compute GAGS score
        total_gags_score = 0
        zone_summary: Dict[str, Any] = {}
        for z in REGION_NAMES:
            factor = ZONE_FACTORS.get(z, 1)
            max_grade = zone_max_grades.get(z, 0)
            score = max_grade * factor
            total_gags_score += score
            zone_summary[z] = {
                "count": zone_counts.get(z, 0),
                "max_grade": max_grade,
                "factor": factor,
                "score": score,
                "pixel_count": pixel_counts.get(z, 0),
                "surface_percentage": pixel_percentages.get(z, 0.0),
            }

        if not filter_non_facial:
            zone_summary["unassigned"] = {
                "count": zone_counts.get("unassigned", 0),
                "max_grade": zone_max_grades.get("unassigned", 0),
                "factor": 0,
                "score": 0,
                "pixel_count": 0,
                "surface_percentage": 0.0,
            }

        severity_title, severity_desc = get_gags_severity(total_gags_score)
        acne_penalty = round(min(total_gags_score / 32.0, 1.0) * 35.0, 2)

        return {
            "image_width": img_w,
            "image_height": img_h,
            "confidence_threshold": detection_result.get("confidence_threshold", 0.15),
            "filter_non_facial": filter_non_facial,
            "non_facial_count": len(excluded_non_facial),
            "raw_detections_count": len(raw_detections),
            "orientation": {
                "view": view,
                "metrics": orientation_metrics,
            },
            "regions": {
                "pixel_counts": pixel_counts,
                "pixel_percentages": pixel_percentages,
            },
            "detections": enriched_detections,
            "total_detections": len(enriched_detections),
            "lesions_summary": lesions_summary,
            "zone_summary": zone_summary,
            "gags": {
                "total_score": total_gags_score,
                "max_facial_score": 32,
                "max_score": 32,
                "acne_penalty": acne_penalty,
                "severity": severity_title,
                "description": severity_desc,
            },
            "visualizations": visualizations,
            "v3_preprocessing": {
                "git_branch": v3_result["v3_metadata"]["branch"],
                "git_commit": v3_result["v3_metadata"]["commit_sha"],
                "module_path": v3_result["v3_metadata"]["module_path"],
                "model_input_confirmed": True,
                "model_input_dimensions": [cut_w, cut_h],
                "model_input_origin": [int(ox), int(oy)],
                "candidate_count": v3_result["v3_metadata"]["candidate_count"],
                "skin_fallback": v3_result["skin_fallback"],
                "stage_traces": v3_result["stage_traces"],
            },
        }
