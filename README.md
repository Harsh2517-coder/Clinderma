# Pigmentation Detection

Detects and segments pigmentation regions on facial skin with a trained Mask R-CNN, behind a Streamlit page. Part of Clinderma (branch **pigmentation**). Research/demo project, not a clinically validated diagnostic system.

## Model

| Item | Value |
|---|---|
| Model | Mask R-CNN (Mask Region-based Convolutional Neural Network) |
| Backbone | ResNet-50 |
| Feature extraction | Feature Pyramid Network (FPN) |
| Task | Detection and instance segmentation of pigmentation regions on facial skin |
| Output | Predicted pigmentation regions with segmentation masks and confidence scores |

## Framework and technologies

- **Deep learning framework:** PyTorch
- **Computer vision library:** Torchvision (Mask R-CNN ResNet-50 FPN builders)
- **Language:** Python
- **User interface:** Streamlit
- **Model checkpoint:** `best_model.pth` (168.06 MB, stored with Git LFS)

## Trained weights: use without training

`best_model.pth` holds the **trained model's weights**, so the model can be used directly for inference without training it again. Install Git LFS before cloning (otherwise you only get a small pointer file instead of the weights):

```cmd
git lfs install
git clone -b pigmentation https://github.com/Harsh2517-coder/Clinderma.git
```

Inference settings are in `inference_config.json`: confidence threshold 0.75, CLAHE off, horizontal-flip test-time augmentation on. All three can be changed in the app's sidebar.

## Where it runs now

The model currently runs **locally on a Windows laptop**; nothing is sent to a cloud service.

1. The trained model is loaded locally for inference.
2. Streamlit is the front end: the user uploads a face image (jpg, jpeg, png or webp).
3. The model processes the image and the page shows the detected pigmentation regions and their segmentation masks.

What the app does with an uploaded image: optional CLAHE contrast fix (automatic when the image is low-contrast) -> Mask R-CNN -> optional flip test-time augmentation (normal + flipped results merged with NMS) -> confidence-threshold slider -> optional restriction to the facial skin zones (`generate_face_regions_v2.py`) -> mask overlay, the five anatomical region masks, and the number of detections. If a ground-truth mask is uploaded it also reports Dice and IoU.

## Dataset

A pigmentation-specific dataset was not supplied with the project. Publicly available Roboflow dataset material was used to prepare approximately 1,000 images for the pigmentation workflow, and the images and annotations were used to train the model. Only redistribute images when their licenses permit it.

## Files

```text
streamlit_app.py                         the app
generate_face_regions_v2.py              face-region helper imported by the app (keep it beside streamlit_app.py)
inference_config.json                    threshold, CLAHE and TTA defaults
best_model.pth                           trained weights (Git LFS)
pigmentation_maskrcnn_v2_improved (1).ipynb   training notebook
requirements.txt
preprocessing/, acne/, docs/, examples/  shared with the other branches
```

## Requirements and installation (Windows Command Prompt)

Use a dedicated Python virtual environment (Python 3.10 is a sensible starting point; confirm the version your PyTorch and Torchvision install supports).

```cmd
cd Clinderma
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch and Torchvision must be compatible with each other and with your system. For GPU support, install the matching CUDA build from the official PyTorch instructions first, then the remaining requirements.

## Run

```cmd
streamlit run streamlit_app.py
```

Upload a face image, adjust the confidence threshold and options in the sidebar, and inspect the predicted regions. If the app reports a missing checkpoint or file, check that `best_model.pth` is a real file (not an LFS pointer) and that `inference_config.json` and `generate_face_regions_v2.py` sit beside `streamlit_app.py`.

## Evaluation and limitations

- Segmentation quality is measured with Intersection over Union (IoU) and Dice against ground-truth masks on a labelled validation/test set. Do not report a score unless it was actually calculated on the relevant labelled data. A model confidence score is not a clinical probability or a diagnosis.
- Results depend on dataset quality, annotations, lighting, pose, skin tone and image quality. The model is for research and demonstration, not medical diagnosis.

## Future work

- Evaluate on a held-out labelled test set and document the measured metrics.
- Check performance across skin tones, poses and lighting.
- Document the final dataset split and model version.

## Other branches

`main` (five-zone preprocessing), `V2`, `V3-preprocess-training`, `patient-preprocessed-first5`, `patient-preprocessed-all`, `tanaya_prototype` (acne + GAGS prototype), `face-wellness-platform` (FaceVeda, training, 3D face map).
