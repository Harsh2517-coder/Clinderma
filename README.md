# Pigmentation Detection

A focused project for detecting and segmenting pigmentation regions in facial images using a trained Mask R-CNN model and a Streamlit interface.

## Overview

The application allows a user to upload a facial image, run the trained model, and view the predicted pigmentation regions. A confidence threshold control filters which detections are displayed.

This is a research/demo project, not a clinically validated diagnostic system.

## Dataset

A pigmentation-specific dataset was not supplied with the project. Publicly available Roboflow dataset material was used to prepare approximately 1,000 images for the pigmentation workflow.

The images and annotations were used to train the model. Update the dataset split counts here if they change in the final training setup. Only redistribute images when their licenses permit it.

## Model

- **Task:** Pigmentation-region detection and segmentation
- **Architecture:** Mask R-CNN
- **Backbone:** ResNet-50 with Feature Pyramid Network (FPN)
- **Framework:** PyTorch and Torchvision
- **Interface:** Streamlit
- **Inference configuration:** `inference_config.json`, if used by the application
- **Trained weights:** `best_model.pth` (must be available at the path expected by the code)

The model produces candidate regions, confidence scores, and segmentation masks. The confidence threshold controls which detections are displayed; changing it does not retrain the model.

## Project files

```text
Clinderma/
├── streamlit_app.py
├── pigmentation_maskrcnn_v2_improved.py
├── inference_config.json
├── best_model.pth                 # model weights; may be stored separately
├── generate_face_regions_v2.py    # include/use only if the app imports it
├── requirements.txt
└── README.md
```

The actual repository may contain other files. This README focuses on the pigmentation module only.

## Requirements

Use a dedicated Python virtual environment. Python 3.10 is a sensible starting point for compatibility with the existing project, but confirm the Python version supported by the installed PyTorch, Torchvision, and other packages on your machine.

The `requirements.txt` file lists the Python packages used by the app. PyTorch and Torchvision should be compatible with each other and with your operating system. If you need CUDA/GPU support, install the appropriate PyTorch build for your system.

## Installation (Windows Command Prompt)

Clone the repository and switch to the pigmentation branch:

```cmd
git clone -b pigmentation https://github.com/Harsh2517-coder/Clinderma.git
cd Clinderma
```

Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Upgrade pip:

```cmd
python -m pip install --upgrade pip
```

Install dependencies:

```cmd
python -m pip install -r requirements.txt
```

If PyTorch installation fails or you need GPU support, install a compatible PyTorch/Torchvision pair using the official PyTorch installation instructions, then install the remaining requirements.

## Model checkpoint

The application needs the trained model checkpoint. Make sure `best_model.pth` is available at the path referenced by `streamlit_app.py` and the model-loading code.

If the checkpoint is too large for normal GitHub storage, keep it in Git LFS or suitable external storage and document how to retrieve it. Do not commit secrets or private images.

## Run the application

From the repository root:

```cmd
streamlit run streamlit_app.py
```

Then:

1. Upload a supported facial image.
2. Run the prediction.
3. Inspect the predicted pigmentation regions.
4. Adjust the confidence threshold to change which detections are displayed.

If the app reports a missing checkpoint or configuration file, check the paths used in the code.

## Evaluation

Evaluate segmentation using a labeled validation/test set. Metrics such as Intersection over Union (IoU) and Dice score compare predicted masks with ground-truth masks. Do not report a score unless it has actually been calculated on the relevant labeled data.

A model confidence score is not a clinical probability or diagnosis.

## Limitations

- Results depend on dataset quality, annotations, lighting, pose, skin tone, and image quality.
- The model has not been established as a clinically validated diagnostic tool.
- The application should be used for research and demonstration, not medical diagnosis.

## Future work

- Evaluate on a held-out labeled test set and document measured metrics.
- Verify performance across different skin tones, poses, and lighting conditions.
- Improve user-facing error handling and checkpoint setup.
- Document the final dataset split and model version.
