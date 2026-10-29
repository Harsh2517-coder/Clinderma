import io
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Keep these files beside this app:
# best_model.pth, inference_config.json, generate_face_regions_v2.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CONFIG_PATH = BASE_DIR / "inference_config.json"

st.set_page_config(page_title="Clinderma | Pigmentation Analysis", page_icon="🔬", layout="wide")


def apply_clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def assess_image_quality(rgb: np.ndarray) -> dict:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]
    brightness = float(np.mean(luminance)) / 255.0
    contrast = float(np.std(luminance)) / 255.0
    bright_clipped = float(np.mean(luminance >= 250))
    dark_clipped = float(np.mean(luminance <= 5))
    severe_overexposure = bright_clipped >= 0.10
    low_contrast = contrast < 0.16
    apply_automatically = low_contrast and not severe_overexposure

    if severe_overexposure:
        message = "A noticeable part of the image is nearly pure white. CLAHE cannot recover detail lost to overexposure."
    elif low_contrast:
        message = "Low contrast detected. CLAHE is recommended for this image."
    else:
        message = "Contrast looks acceptable. Automatic CLAHE is not needed."

    return {
        "brightness": brightness,
        "contrast": contrast,
        "bright_clipped": bright_clipped,
        "dark_clipped": dark_clipped,
        "low_contrast": low_contrast,
        "severe_overexposure": severe_overexposure,
        "apply_automatically": apply_automatically,
        "message": message,
    }


def build_model_from_checkpoint_config(cfg: dict):
    use_v2 = bool(cfg["use_v2"])
    anchor_sizes = tuple(int(x) for x in cfg["anchor_sizes"])
    min_sizes = tuple(int(x) for x in cfg["min_sizes"])
    max_size = int(cfg["max_size"])
    num_classes = int(cfg["num_classes"])
    kwargs = {"min_size": min_sizes, "max_size": max_size}

    if use_v2:
        model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **kwargs)
    else:
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, **kwargs)

    model.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple((s,) for s in anchor_sizes),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, num_classes)
    return model


@st.cache_resource(show_spinner="Loading your saved Mask R-CNN model…")
def load_model(model_path: str, device_name: str):
    device = torch.device(device_name)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint format mismatch: expected keys 'model' and 'config'.")
    model = build_model_from_checkpoint_config(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def predict_with_optional_flip_tta(model, image_tensor, use_tta: bool):
    with torch.inference_mode():
        prediction = model([image_tensor])[0]
        if not use_tta:
            return prediction

        flipped_prediction = model([image_tensor.flip(-1)])[0]
        width = image_tensor.shape[-1]
        flipped_boxes = flipped_prediction["boxes"].clone()
        flipped_boxes[:, [0, 2]] = width - flipped_prediction["boxes"][:, [2, 0]]
        boxes = torch.cat([prediction["boxes"], flipped_boxes])
        scores = torch.cat([prediction["scores"], flipped_prediction["scores"]])
        labels = torch.cat([prediction["labels"], flipped_prediction["labels"]])
        masks = torch.cat([prediction["masks"], flipped_prediction["masks"].flip(-1)])
        keep = torchvision.ops.nms(boxes, scores, 0.5)
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
            "masks": masks[keep],
        }


@st.cache_data(show_spinner="Generating Clinderma facial regions…")
def generate_clinderma_regions(image_bytes: bytes):
    """
    Use the repository's v2 pipeline (MediaPipe landmarks + semantic parsing)
    and return five anatomical masks. The union is used only to restrict displayed
    pigmentation predictions; the input image to Mask R-CNN is not blacked out.
    """
    # The helper module must be alongside streamlit_app.py.
    from generate_face_regions_v2 import (
        BoundaryConfig,
        OrientationThresholds,
        build_region_masks,
        classify_view,
        detect_face_landmarks,
        face_oval_mask,
        landmarks_to_pixels,
        semantic_face_parse,
    )

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.asarray(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    landmarks = detect_face_landmarks(bgr, min_detection_confidence=0.5)
    h, w = bgr.shape[:2]
    points = landmarks_to_pixels(landmarks, w, h)
    oval = face_oval_mask((h, w), points)
    view, metrics = classify_view(points, oval, OrientationThresholds())
    parsing_labels, id2label = semantic_face_parse(bgr, "jonathandinu/face-parsing")
    regions = build_region_masks(
        bgr.shape, points, parsing_labels, id2label, BoundaryConfig(), view
    )
    union = np.zeros((h, w), dtype=np.uint8)
    for mask in regions.values():
        union = cv2.bitwise_or(union, mask)
    return union > 0, {name: mask for name, mask in regions.items()}, view.value, metrics


def overlay_masks(rgb: np.ndarray, masks: np.ndarray, opacity: float):
    result = rgb.astype(np.float32).copy()
    tint = np.array([255.0, 45.0, 65.0], dtype=np.float32)
    for mask in masks:
        if mask.any():
            result[mask] = (1.0 - opacity) * result[mask] + opacity * tint
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def calculate_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    if pred.shape != gt.shape:
        raise ValueError("Ground-truth mask dimensions must match the uploaded image.")
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()
    dice = 2 * intersection / denominator if denominator else 1.0
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union else 1.0
    return float(dice), float(iou)


st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
div[data-testid="stAppDeployButton"] {display: none !important;}
.hero {padding: 1.2rem 1.4rem; border-radius: 18px;
       background: linear-gradient(120deg, rgba(130,100,220,.18), rgba(235,145,170,.13));
       border: 1px solid rgba(130,100,220,.25); margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)
st.title("🔬 Clinderma")
st.caption("AI-assisted pigmentation segmentation · Mask R-CNN")
st.markdown(
    '<div class="hero"><h3 style="margin:0">Pigmentation Analysis</h3>'
    '<p style="margin:.45rem 0 0 0">Uses Clinderma v2 facial-region preprocessing to limit displayed predictions to detected facial regions.</p></div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Place best_model.pth beside streamlit_app.py.")
    st.stop()
if not CONFIG_PATH.exists():
    st.error(f"Configuration file not found: {CONFIG_PATH}")
    st.info("Place inference_config.json beside streamlit_app.py.")
    st.stop()
if not (BASE_DIR / "generate_face_regions_v2.py").exists():
    st.error("Missing generate_face_regions_v2.py. Place the repository's v2 preprocessing file beside streamlit_app.py.")
    st.stop()

try:
    inference_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    st.error(f"Could not read inference_config.json: {exc}")
    st.stop()

device_name = "cuda" if torch.cuda.is_available() else "cpu"
default_threshold = float(inference_config.get("threshold", 0.75))

with st.sidebar:
    st.header("Prediction settings")
    clahe_mode = st.radio(
        "Image contrast handling",
        ["Auto (recommended)", "Always apply CLAHE", "Never apply CLAHE"],
        index=0,
        help="Auto applies CLAHE only when contrast is low and severe overexposure is not detected.",
    )
    use_tta = st.checkbox(
        "Horizontal-flip test-time augmentation (TTA)",
        value=bool(inference_config.get("tta", True)),
        help="Runs normal and flipped inference and merges detections with NMS.",
    )
    restrict_to_face = st.checkbox(
        "Restrict predictions to Clinderma facial regions",
        value=True,
        help="Uses your repository's v2 face parser to keep predicted pixels only inside forehead, cheeks, nose, and chin masks.",
    )
    show_regions = st.checkbox("Show five anatomical region masks", value=True)
    st.caption(f"Device: {device_name.upper()}")
    st.caption("Saved Mask R-CNN checkpoint is loaded once and reused.")

confidence = st.slider(
    "Confidence threshold", 0.05, 0.95,
    value=min(0.95, max(0.05, default_threshold)), step=0.05, format="%.2f",
)
st.caption(f"Only detections with confidence ≥ {confidence:.0%} will be displayed.")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is None:
    st.info("Upload a JPG, PNG, or WEBP image to begin.")
    st.stop()

image_bytes = uploaded_file.getvalue()
try:
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_rgb = np.asarray(original_image)
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

facial_mask = None
region_masks = {}
view_name = None
orientation_metrics = {}
if restrict_to_face:
    try:
        facial_mask, region_masks, view_name, orientation_metrics = generate_clinderma_regions(image_bytes)
        if facial_mask.shape != original_rgb.shape[:2]:
            raise ValueError("Preprocessing mask dimensions do not match the uploaded image.")
    except Exception as exc:
        st.error("Clinderma v2 facial preprocessing failed.")
        st.exception(exc)
        st.info("You can temporarily disable facial-region restriction to run the original Mask R-CNN prediction.")
        st.stop()

try:
    model, checkpoint = load_model(str(MODEL_PATH), device_name)
    device = torch.device(device_name)
    quality = assess_image_quality(original_rgb)
    if clahe_mode == "Always apply CLAHE":
        clahe_applied = True
    elif clahe_mode == "Never apply CLAHE":
        clahe_applied = False
    else:
        clahe_applied = bool(quality["apply_automatically"])

    input_rgb = apply_clahe(original_rgb) if clahe_applied else original_rgb
    image_tensor = torch.from_numpy(np.ascontiguousarray(input_rgb)).permute(2, 0, 1).float().div(255.0).to(device)
    prediction = predict_with_optional_flip_tta(model, image_tensor, use_tta)
except Exception as exc:
    st.error("Model loading or prediction failed.")
    st.exception(exc)
    st.stop()

scores_all = prediction["scores"].detach().cpu().numpy()
keep = scores_all >= confidence
scores = scores_all[keep]
raw_masks = (prediction["masks"][keep, 0] > 0.5).detach().cpu().numpy()
boxes = prediction["boxes"][keep].detach().cpu().numpy()

if restrict_to_face and facial_mask is not None:
    masks = raw_masks & facial_mask[None, :, :]
else:
    masks = raw_masks

st.subheader("Image quality check")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Mean brightness", f"{quality['brightness'] * 100:.1f}%")
q2.metric("Contrast estimate", f"{quality['contrast'] * 100:.1f}%")
q3.metric("Near-white pixels", f"{quality['bright_clipped'] * 100:.2f}%")
q4.metric("CLAHE preprocessing", "Applied" if clahe_applied else "Not applied")
st.caption(quality["message"])
if quality["severe_overexposure"]:
    st.warning("The image may be overexposed. CLAHE cannot recover clipped highlights.")
elif clahe_applied:
    st.info("CLAHE was applied before inference. Results are overlaid on the original image.")

if restrict_to_face and facial_mask is not None:
    st.subheader("Clinderma facial-region preprocessing (v2)")
    st.caption(f"Estimated view: {view_name}. These masks are anatomical preprocessing regions, not a clinical diagnosis.")
    if orientation_metrics:
        a, b = st.columns(2)
        a.metric("Nose shift ratio", f"{orientation_metrics.get('nose_shift_ratio', 0):.3f}")
        b.metric("Eye-width ratio", f"{orientation_metrics.get('eye_width_ratio', 0):.3f}")
    preview_cols = st.columns(3)
    with preview_cols[0]:
        st.image(original_image, caption="Original image", use_container_width=True)
    with preview_cols[1]:
        st.image(facial_mask.astype(np.uint8) * 255, caption="Union of Clinderma facial regions", clamp=True, use_container_width=True)
    with preview_cols[2]:
        st.image(overlay_masks(original_rgb, [facial_mask], 0.25), caption="Region boundary preview", use_container_width=True)
    if show_regions:
        region_cols = st.columns(5)
        for col, (name, mask) in zip(region_cols, region_masks.items()):
            with col:
                st.image(mask, caption=name.replace("_", " ").title(), clamp=True, use_container_width=True)

left, middle, right = st.columns(3, gap="large")
with left:
    st.subheader("Original image")
    st.image(original_image, use_container_width=True)
with middle:
    st.subheader("Image sent to model")
    st.image(input_rgb, use_container_width=True)
    st.caption("CLAHE-enhanced input" if clahe_applied else "Original pixels; no CLAHE")
with right:
    st.subheader("Pigmentation segmentation")
    if len(masks) and any(mask.any() for mask in masks):
        st.image(overlay_masks(original_rgb, masks, 0.38), use_container_width=True)
    else:
        st.image(original_image, use_container_width=True)
        st.warning("No regions passed the confidence threshold and facial-region restriction.")

st.subheader("Prediction summary")
m1, m2, m3 = st.columns(3)
m1.metric("Detections above threshold", int(len(scores)))
m2.metric("Mean confidence", f"{float(scores.mean() * 100):.1f}%" if len(scores) else "—")
m3.metric("Model best validation F1", f"{float(checkpoint['val_f1']):.3f}" if checkpoint.get("val_f1") is not None else "—")
if restrict_to_face:
    st.caption("Detection count and confidence refer to Mask R-CNN detections before facial-mask clipping; the displayed mask pixels are restricted to the Clinderma region union.")

if len(scores):
    rows = []
    for index, (score, box, mask) in enumerate(zip(scores, boxes, masks), start=1):
        rows.append({
            "Region": index,
            "Confidence": f"{score * 100:.1f}%",
            "Displayed mask pixels": int(mask.sum()),
            "Bounding box (x1, y1, x2, y2)": ", ".join(str(int(v)) for v in box),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    overlay = overlay_masks(original_rgb, masks, 0.38)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    st.download_button("Download segmentation overlay", buffer.getvalue(), "clinderma_pigmentation_overlay.png", "image/png")

st.divider()
st.subheader("Optional: evaluate against a ground-truth mask")
st.write("Upload a labeled pigmentation mask for this exact image (pigmentation pixels white, background black). Dice/IoU here are single-image metrics.")
gt_upload = st.file_uploader("Upload ground-truth mask (optional)", type=["png", "jpg", "jpeg"], key="ground_truth_mask")
if gt_upload is not None:
    try:
        gt_image = Image.open(gt_upload).convert("L")
        if gt_image.size != original_image.size:
            st.error(f"Ground-truth mask size {gt_image.size} does not match uploaded image size {original_image.size}.")
        else:
            gt_mask = np.asarray(gt_image) > 127
            combined_pred = masks.any(axis=0) if len(masks) else np.zeros(original_rgb.shape[:2], dtype=bool)
            intersection = np.logical_and(combined_pred, gt_mask).sum()
            denominator = combined_pred.sum() + gt_mask.sum()
            dice = (2 * intersection / denominator) if denominator else 1.0
            union = np.logical_or(combined_pred, gt_mask).sum()
            iou = intersection / union if union else 1.0
            c1, c2 = st.columns(2)
            c1.metric("Dice score", f"{dice:.4f}")
            c2.metric("IoU", f"{iou:.4f}")
    except Exception as exc:
        st.error(f"Could not evaluate the ground-truth mask: {exc}")

with st.expander("Checkpoint details"):
    st.write(f"Checkpoint: `{MODEL_PATH.name}`")
    st.write(f"Best epoch (1-indexed): {int(checkpoint.get('epoch', -1)) + 1}")
    st.write(f"Saved best validation F1: {checkpoint.get('val_f1', 'not present')}")
    st.json(checkpoint.get("config", {}))
