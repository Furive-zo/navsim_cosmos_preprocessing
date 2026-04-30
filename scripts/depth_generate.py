"""
Depth Anything V2 → 255-normalized grayscale depth map

흐름:
  Depth Anything V2 (HuggingFace) → relative depth → [0, 255] uint8 PNG
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image as PILImage


def build_depth_model(model_id, device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def run_depth(image_rgb, processor, model, device):
    """Returns (H, W) uint8 depth map normalized to [0, 255]."""
    H, W = image_rgb.shape[:2]
    pil  = PILImage.fromarray(image_rgb)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth  # [1, h', w']
    depth_up = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()

    d_min, d_max = depth_up.min(), depth_up.max()
    if d_max - d_min < 1e-6:
        return np.zeros((H, W), dtype=np.uint8)
    normalized = (depth_up - d_min) / (d_max - d_min) * 255.0
    return normalized.astype(np.uint8)


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Depth Anything V2 → 255-normalized depth map")
    p.add_argument("--input_dir",  default=os.path.join(script_dir, "..", "dataset", "CAM_F0"))
    p.add_argument("--output_dir", default=os.path.join(script_dir, "..", "dataset", "CAM_F0_depth"))
    p.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="HuggingFace model ID (Small/Base/Large-hf)",
    )
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading Depth Anything V2 ({args.model})...")
    processor, model = build_depth_model(args.model, device)

    os.makedirs(args.output_dir, exist_ok=True)

    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(f for f in os.listdir(args.input_dir)
                   if os.path.splitext(f)[1].lower() in exts)
    assert files, f"No images in {args.input_dir}"
    print(f"Processing {len(files)} images...")

    for i, fname in enumerate(files):
        img = cv2.imread(os.path.join(args.input_dir, fname))
        assert img is not None
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = run_depth(image_rgb, processor, model, device)

        out_name = os.path.splitext(fname)[0] + ".png"
        cv2.imwrite(os.path.join(args.output_dir, out_name), depth)

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {fname}")

    print(f"\nDone → {args.output_dir}")


if __name__ == "__main__":
    main()
