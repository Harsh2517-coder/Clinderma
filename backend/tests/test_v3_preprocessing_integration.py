"""Comprehensive Test Suite for Clinderma V3 Preprocessing Pipeline Integration.

Verifies:
  1. The application loads the authoritative V3 preprocessing implementation from V_3.
  2. The Git branch and commit SHA match the authoritative V3 commit.
  3. Every required preprocessing stage executes in the defined order without skipping.
  4. The model strictly receives the preprocessed face cutout output, NOT the raw image.
  5. The raw image cannot bypass preprocessing during normal inference or edge cases.
  6. Non-facial / invalid images fail explicitly without silently falling back to the raw image.
  7. The existing acne detection model functions correctly and returns valid detections.
  8. Bounding-box coordinates remain consistent throughout the pipeline.
"""

import os
import sys
import io
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from fastapi.testclient import TestClient

# Set up paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
V3_DIR = REPO_ROOT / "preprocessing" / "Clinderma" / "V_3"
SAMPLE_IMG_PATH = REPO_ROOT / "preprocessing" / "Clinderma" / "examples" / "input" / "levle1_340.jpg"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.v3_preprocessor import V3Preprocessor, get_v3_git_info, V3_MODULE_FILE
from app.pipeline import ClindermaPipeline
from app.main import app, pipeline
from app.acne_detector import AcneDetector


@pytest.fixture(scope="session")
def representative_image_bytes():
    assert SAMPLE_IMG_PATH.exists(), f"Sample image not found at {SAMPLE_IMG_PATH}"
    with open(SAMPLE_IMG_PATH, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def representative_image_bgr():
    img = cv2.imread(str(SAMPLE_IMG_PATH))
    assert img is not None, "Failed to load sample image with cv2"
    return img


@pytest.fixture
def client():
    return TestClient(app)


class TestV3ModuleLoadingAndGitMetadata:
    """Verifies that the application loads the intended V3 preprocessing module and git commit."""

    def test_v3_module_path(self):
        preprocessor = V3Preprocessor()
        assert Path(preprocessor.source_path).resolve() == V3_DIR.resolve()
        assert Path(preprocessor.module_file).name == "preprocess.py"
        assert Path(preprocessor.module_file).exists()

    def test_v3_git_metadata(self):
        git_info = get_v3_git_info()
        assert git_info["branch"] in ("v3", "V3-preprocess-training")
        assert len(git_info["commit_sha"]) == 40
        # Authoritative V3 commit SHA
        assert git_info["commit_sha"].startswith("ef630bf")

    def test_health_endpoint_reports_v3_metadata(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "v3_preprocessing" in data
        v3_meta = data["v3_preprocessing"]
        assert v3_meta["integrated"] is True
        assert v3_meta["git_branch"] in ("v3", "V3-preprocess-training")
        assert v3_meta["git_commit"].startswith("ef630bf")


class TestV3PipelineStagesExecution:
    """Verifies that all required V3 preprocessing stages execute in exact order with valid tensor types."""

    def test_all_stages_executed_in_order(self, representative_image_bgr):
        preprocessor = V3Preprocessor()
        result = preprocessor.process(representative_image_bgr)

        assert "stage_traces" in result
        traces = result["stage_traces"]

        stage_numbers = [t["stage_number"] for t in traces]
        assert stage_numbers == [0, 1, 2, 3, 4, 5], f"Unexpected stage execution sequence: {stage_numbers}"

        for trace in traces:
            assert trace["status"] == "SUCCESS", f"Stage {trace['stage_name']} failed: {trace['details']}"
            assert trace["elapsed_ms"] >= 0

    def test_intermediate_tensors_and_cutout_properties(self, representative_image_bgr):
        preprocessor = V3Preprocessor()
        result = preprocessor.process(representative_image_bgr)

        # Stage 5 output: preprocessed face cutout
        cutout = result["cutout_image"]
        assert isinstance(cutout, np.ndarray)
        assert cutout.ndim == 3 and cutout.shape[2] == 3
        assert cutout.dtype == np.uint8

        # Origin coordinates
        ox, oy = result["cutout_origin"]
        assert isinstance(ox, int) and isinstance(oy, int)
        assert ox >= 0 and oy >= 0

        # Cutout dimensions should be smaller than or equal to original image
        h_orig, w_orig = representative_image_bgr.shape[:2]
        h_cut, w_cut = cutout.shape[:2]
        assert h_cut <= h_orig and w_cut <= w_orig

        # Candidates generated
        assert len(result["candidate_boxes_full"]) > 0
        assert len(result["candidates"]) > 0

        # Stages images present
        stages = result["stages_bgr"]
        for key in ("1_original", "2_clahe", "3_face_skin_mask", "4_candidate_mask", "5_final_boxes", "6_model_input_cutout"):
            assert key in stages
            assert isinstance(stages[key], np.ndarray)


class TestModelStrictlyConsumesPreprocessedCutout:
    """Verifies that the acne detection model receives ONLY the V3 preprocessed cutout, NEVER the raw image."""

    def test_detector_receives_exact_v3_cutout(self, representative_image_bytes):
        pipeline_instance = ClindermaPipeline()
        received_payloads = []

        # Spy on detector.predict to capture exact bytes passed
        original_predict = pipeline_instance.detector.predict

        def mock_predict(image_bytes, **kwargs):
            received_payloads.append(image_bytes)
            return {
                "image_width": 500,
                "image_height": 500,
                "detections": [
                    {
                        "class": "pustule",
                        "confidence": 0.85,
                        "bbox": {"x1": 500.0, "y1": 500.0, "x2": 550.0, "y2": 560.0},
                    }
                ],
                "total_detections": 1,
                "lesions_summary": {"pustule": 1},
                "confidence_threshold": 0.15,
            }

        pipeline_instance.detector.predict = mock_predict

        result = pipeline_instance.run(representative_image_bytes, filter_non_facial=False)

        # Verify detector was called exactly once
        assert len(received_payloads) == 1
        model_input_bytes = received_payloads[0]

        # Verify model input is NOT the original raw image bytes
        assert model_input_bytes != representative_image_bytes, "CRITICAL: Model received the raw image!"

        # Decode model input and verify it matches the preprocessed face cutout dimensions
        model_input_img = cv2.imdecode(np.frombuffer(model_input_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert model_input_img is not None
        v3_meta = result["v3_preprocessing"]
        assert list(model_input_img.shape[:2]) == [v3_meta["model_input_dimensions"][1], v3_meta["model_input_dimensions"][0]]
        assert v3_meta["model_input_confirmed"] is True

        # Verify bounding box was mapped back to original coordinates
        det = result["detections"][0]
        ox, oy = v3_meta["model_input_origin"]
        assert det["bbox"]["x1"] == 500.0 + ox
        assert det["bbox"]["y1"] == 500.0 + oy
        assert det["bbox"]["x2"] == 550.0 + ox
        assert det["bbox"]["y2"] == 560.0 + oy

    def test_analyze_endpoint_enforces_v3_preprocessing(self, client, representative_image_bytes):
        """Verifies that /api/acne/analyze also enforces V3 preprocessing and does NOT pass raw image to model."""
        with patch.object(pipeline.detector, "predict") as mock_det:
            mock_det.return_value = {
                "detections": [],
                "total_detections": 0,
                "lesions_summary": {},
                "confidence_threshold": 0.15,
            }
            files = {"file": ("test.jpg", representative_image_bytes, "image/jpeg")}
            res = client.post("/api/acne/analyze", files=files)
            assert res.status_code == 200
            data = res.json()

            # Confirm detector received cutout bytes, not raw image
            call_args = mock_det.call_args[0]
            passed_bytes = call_args[0]
            assert passed_bytes != representative_image_bytes
            assert data["v3_preprocessing"]["consumed_by_model"] is True


class TestErrorHandlingAndNoFallbackToRaw:
    """Verifies that invalid or non-facial images fail explicitly and NEVER silently fall back to raw image."""

    def test_non_facial_image_rejected_without_fallback(self, client):
        # Create synthetic blank / landscape image without a face
        blank_image = np.zeros((400, 400, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank_image)
        blank_bytes = buf.tobytes()

        # Call pipeline run endpoint
        files = {"file": ("blank.jpg", blank_bytes, "image/jpeg")}
        res = client.post("/api/pipeline/run", files=files)

        # Must return 422 Unprocessable Entity with descriptive error
        assert res.status_code == 422
        assert "No human face detected" in res.json()["detail"]

    def test_corrupted_image_file_rejected(self, client):
        corrupted_bytes = b"NOT_A_VALID_JPEG_IMAGE_DATA_12345"
        files = {"file": ("corrupt.jpg", corrupted_bytes, "image/jpeg")}
        res = client.post("/api/pipeline/run", files=files)
        assert res.status_code == 422
        assert "Could not decode image" in res.json()["detail"]


class TestEndToEndPipelineWithTrainedModel:
    """Runs a complete end-to-end integration test with the actual trained model."""

    def test_full_pipeline_end_to_end(self, representative_image_bytes):
        pipeline = ClindermaPipeline()
        result = pipeline.run(representative_image_bytes, confidence=0.15)

        # Check top-level result contract
        assert "image_width" in result
        assert "image_height" in result
        assert "detections" in result
        assert "total_detections" in result
        assert "zone_summary" in result
        assert "gags" in result
        assert "visualizations" in result
        assert "v3_preprocessing" in result

        # Check V3 metadata
        v3 = result["v3_preprocessing"]
        assert v3["git_branch"] in ("v3", "V3-preprocess-training")
        assert v3["git_commit"].startswith("ef630bf")
        assert v3["model_input_confirmed"] is True
        assert len(v3["stage_traces"]) >= 5

        # Check bounding box integrity
        w, h = result["image_width"], result["image_height"]
        for det in result["detections"]:
            b = det["bbox"]
            assert 0 <= b["x1"] < b["x2"] <= w, f"Invalid bbox x coords: {b}"
            assert 0 <= b["y1"] < b["y2"] <= h, f"Invalid bbox y coords: {b}"
            assert det["zone"] in ("forehead", "left_cheek", "right_cheek", "nose", "chin", "unassigned")
            assert det["gags_grade"] in (1, 2, 3, 4)

        # Check GAGS calculation
        gags = result["gags"]
        assert gags["total_score"] >= 0
        assert gags["max_facial_score"] == 32
        assert gags["severity"] in ("Clear", "Mild", "Moderate", "Severe", "Very Severe")
