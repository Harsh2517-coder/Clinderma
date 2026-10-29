"""Frontend: capture (webcam) or upload a face photo ->
  1. face outline with 0% background (hair/neck/clothing/scene removed)
  2. every skin spot (acne, pigmentation, or anything else that isn't smooth
     regular skin) boxed on that same face cutout -- no classification yet,
     just "is this a spot or not"."""
import cv2
import numpy as np
import gradio as gr

from preprocess import preprocess_image, render_debug

MISSING_FACE_MSG = "No face detected -- try a clearer, more front-on or profile shot with better lighting."


def _feathered_cutout(crop_bgr, face_mask, feather=3):
    mask = cv2.GaussianBlur(face_mask, (feather * 2 + 1, feather * 2 + 1), 0) if feather > 0 else face_mask
    bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra


def _tight_crop_to_mask(bgra, face_mask):
    ys, xs = np.where(face_mask > 0)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return bgra[y0:y1, x0:x1]


def run(image):
    if image is None:
        return None, None, "", None
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = preprocess_image(img_bgr)  # CLAHE -> face -> skin mask -> candidates (see preprocess.py)
    if result is None:
        return None, None, MISSING_FACE_MSG, None

    # display uses the user's own pixels (crop_original); detection ran on the CLAHE'd crop
    crop, face_mask, boxes = result["crop_original"], result["face_mask"], result["boxes"]
    stages = render_debug(result, img_bgr)
    boxed_crop = stages["5_final_boxes"]

    cutout = _tight_crop_to_mask(_feathered_cutout(crop, face_mask), face_mask)
    boxed_cutout = _tight_crop_to_mask(_feathered_cutout(boxed_crop, face_mask), face_mask)

    cutout_rgba = cv2.cvtColor(cutout, cv2.COLOR_BGRA2RGBA)
    boxed_rgba = cv2.cvtColor(boxed_cutout, cv2.COLOR_BGRA2RGBA)
    gallery = [(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), name) for name, im in stages.items()]
    return cutout_rgba, boxed_rgba, f"{len(boxes)} candidate box(es) -- high-recall proposals, not final detections" + (" [face parsing found little skin: whole crop searched]" if result["skin_fallback"] else ""), gallery


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="numpy", label="Capture or upload a face photo", sources=["webcam", "upload"]),
    outputs=[
        gr.Image(type="numpy", label="Face outline (background removed)", image_mode="RGBA"),
        gr.Image(type="numpy", label="Detected spots", image_mode="RGBA"),
        gr.Textbox(label="Status"),
        gr.Gallery(label="Pipeline stages: original / CLAHE / face+skin mask / candidate mask / final boxes"),
    ],
    title="Face Cropper + Candidate Spot Generator",
    description="Works for front, left-profile, and right-profile shots. CLAHE first, then every plausible skin spot -- acne, pigmentation, or anything else that isn't smooth regular skin -- gets a candidate box (high recall; false positives expected, a downstream model classifies them).",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
