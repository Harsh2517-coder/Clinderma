"""Train RF-DETR small on the local copy of the Roboflow acne dataset (tanayaaa/acne-detection-v1-tjjbs v2, one copy per source image), then export ONNX.
    ./venv/bin/python train.py <epochs> [output_dir]       (re-running resumes from output_dir/checkpoint.pth if it exists)"""
import os, sys
from rfdetr import RFDETRSmall


def main():
    epochs = int(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else "output"
    last = os.path.join(out, "checkpoint.pth")
    extra = {"resume": last} if os.path.exists(last) else {}
    RFDETRSmall().train(dataset_dir="dataset_1x", epochs=epochs, batch_size=2, grad_accum_steps=8, lr=1e-4, output_dir=out, num_workers=0,
                        early_stopping=False, checkpoint_interval=1, progress_bar="tqdm", **extra)


if __name__ == "__main__":   # required on macOS: data-loader workers are spawned
    main()
