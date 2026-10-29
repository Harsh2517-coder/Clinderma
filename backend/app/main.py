import os
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional

from app.pipeline import ClindermaPipeline
from app.v3_preprocessor import bgr_to_base64_jpeg

app = FastAPI(title="Clinderma Unified Acne & Facial Preprocessing API (V3 Preprocessing Integrated)")

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = ClindermaPipeline()

# Samples directory
SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "preprocessing" / "Clinderma" / "examples" / "input"

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    git_info = pipeline.v3_preprocessor.git_info
    return {
        "status": "healthy",
        "model_available": pipeline.detector.is_model_available(),
        "model_id": pipeline.detector.model_id,
        "sample_images_available": SAMPLES_DIR.exists() and len(list(SAMPLES_DIR.glob("*.jpg"))) > 0,
        "v3_preprocessing": {
            "integrated": True,
            "git_branch": git_info["branch"],
            "git_commit": git_info["commit_sha"],
            "source_path": pipeline.v3_preprocessor.source_path,
            "module_file": pipeline.v3_preprocessor.module_file,
        }
    }

@app.get("/api/samples")
async def get_sample_images() -> List[Dict[str, str]]:
    if not SAMPLES_DIR.exists():
        return []
    
    samples = []
    for file_path in sorted(SAMPLES_DIR.glob("*.jpg")):
        name = file_path.name
        label = file_path.stem.replace("_", " ").title()
        samples.append({
            "filename": name,
            "label": label,
            "url": f"/api/samples/{name}"
        })
    return samples

@app.get("/api/samples/{filename}")
async def get_sample_image(filename: str):
    file_path = SAMPLES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(file_path, media_type="image/jpeg")

@app.post("/api/pipeline/run")
async def run_pipeline(
    file: UploadFile = File(...),
    confidence: Optional[float] = None,
    filter_non_facial: bool = True,
    iou_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Runs complete end-to-end pipeline:
    Raw image -> V3 Preprocessing Pipeline -> Preprocessed Cutout -> Acne Model -> Zone Fusion -> GAGS scoring.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Error reading uploaded file")

    try:
        results = pipeline.run(
            contents,
            confidence=confidence,
            filter_non_facial=filter_non_facial,
            iou_threshold=iou_threshold,
        )
        return results
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"Pipeline execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.post("/api/preprocess")
async def preprocess_only(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Runs only the V3 preprocessing pipeline (stages 1-5) and returns candidate proposals and stage previews."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Error reading uploaded file")

    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid or corrupted image")

        v3_res = pipeline.v3_preprocessor.process(img)
        stages_bgr = v3_res.get("stages_bgr", {})
        
        return {
            "v3_metadata": v3_res["v3_metadata"],
            "candidate_count": len(v3_res["candidates"]),
            "candidate_boxes_full": v3_res["candidate_boxes_full"],
            "candidate_boxes_crop": v3_res["candidate_boxes_crop"],
            "skin_fallback": v3_res["skin_fallback"],
            "stage_traces": v3_res["stage_traces"],
            "stages": {
                "1_original": bgr_to_base64_jpeg(stages_bgr.get("1_original", img)),
                "2_clahe": bgr_to_base64_jpeg(stages_bgr.get("2_clahe", img)),
                "3_face_skin_mask": bgr_to_base64_jpeg(stages_bgr.get("3_face_skin_mask", img)),
                "4_candidate_mask": bgr_to_base64_jpeg(stages_bgr.get("4_candidate_mask", img)),
                "5_final_boxes": bgr_to_base64_jpeg(stages_bgr.get("5_final_boxes", img)),
                "6_model_input_cutout": bgr_to_base64_jpeg(v3_res["cutout_image"]),
            }
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/acne/analyze")
async def analyze_acne_only(
    file: UploadFile = File(...),
    confidence: Optional[float] = None,
    iou_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Runs acne detection model with MANDATORY V3 Preprocessing.
    The model STRICTLY consumes the preprocessed face cutout, NEVER the raw image.
    Raw image coordinates are mapped back before returning detections.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Error reading uploaded file")

    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")

        # 1. Mandatory V3 Preprocessing - Extract preprocessed face cutout
        v3_res = pipeline.v3_preprocessor.process(img)
        cutout_bytes = v3_res["cutout_bytes"]
        ox, oy = v3_res["cutout_origin"]

        # 2. Acne detector inference on the preprocessed cutout ONLY
        det_res = pipeline.detector.predict(
            cutout_bytes,
            confidence=confidence,
            iou_threshold=iou_threshold,
        )

        # 3. Map bounding boxes from cutout coordinates back to full image coordinates
        mapped_detections = []
        for det in det_res.get("detections", []):
            cbox = det["bbox"]
            mapped_detections.append({
                "class": det["class"],
                "confidence": det["confidence"],
                "bbox": {
                    "x1": round(cbox["x1"] + ox, 2),
                    "y1": round(cbox["y1"] + oy, 2),
                    "x2": round(cbox["x2"] + ox, 2),
                    "y2": round(cbox["y2"] + oy, 2),
                }
            })

        return {
            "image_width": img.shape[1],
            "image_height": img.shape[0],
            "detections": mapped_detections,
            "total_detections": len(mapped_detections),
            "lesions_summary": det_res.get("lesions_summary", {}),
            "confidence_threshold": det_res.get("confidence_threshold", 0.15),
            "v3_preprocessing": {
                "consumed_by_model": True,
                "git_branch": v3_res["v3_metadata"]["branch"],
                "git_commit": v3_res["v3_metadata"]["commit_sha"],
                "cutout_dimensions": list(v3_res["cutout_image"].shape[:2]),
                "cutout_origin": [int(ox), int(oy)],
                "stage_traces": v3_res["stage_traces"],
            }
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
