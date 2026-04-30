"""
SegFormer (Cityscapes 19-class) → semantic seg map

Cityscapes 팔레트를 그대로 사용하고,
road 영역 내 흰색·노란색 픽셀을 차선으로 추가 구분.
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image as PILImage


# Cityscapes 19-class RGB 팔레트 (index = label ID)
CITYSCAPES_PALETTE = np.array([
    [128,  64, 128],  # 0  road
    [244,  35, 232],  # 1  sidewalk
    [ 70,  70,  70],  # 2  building
    [102, 102, 156],  # 3  wall
    [190, 153, 153],  # 4  fence
    [153, 153, 153],  # 5  pole
    [250, 170,  30],  # 6  traffic light
    [220, 220,   0],  # 7  traffic sign
    [107, 142,  35],  # 8  vegetation
    [152, 251, 152],  # 9  terrain
    [ 70, 130, 180],  # 10 sky
    [220,  20,  60],  # 11 person
    [255,   0,   0],  # 12 rider
    [  0,   0, 142],  # 13 car
    [  0,   0,  70],  # 14 truck
    [  0,  60, 100],  # 15 bus
    [  0,  80, 100],  # 16 train
    [  0,   0, 230],  # 17 motorcycle
    [119,  11,  32],  # 18 bicycle
], dtype=np.uint8)

LANE_COLOR_BGR = (0, 255, 255)   # 차선: 노란색(BGR)


def build_segformer(model_id, device):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def run_segformer(image_rgb, processor, model, device):
    """Returns (H, W) uint8 label map in Cityscapes IDs."""
    H, W  = image_rgb.shape[:2]
    pil   = PILImage.fromarray(image_rgb)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits          # (1, 19, h, w)
    upsampled = torch.nn.functional.interpolate(
        logits, size=(H, W), mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def label_to_color(label_map):
    """label_map (H,W) → Cityscapes 팔레트 기반 BGR seg map."""
    rgb = CITYSCAPES_PALETTE[label_map]   # (H, W, 3) RGB
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def overlay_lanes(image_rgb, label_map, seg_bgr):
    """road/terrain 영역(id 0,1,9) 내 흰색·노란색 픽셀을 차선으로 표시."""
    road_mask = np.isin(label_map, [0, 1, 9])
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    white  = (hsv[:, :, 1] < 50)  & (hsv[:, :, 2] > 180)
    yellow = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35) & \
             (hsv[:, :, 1] > 80)  & (hsv[:, :, 2] > 100)
    seg_bgr[road_mask & (white | yellow)] = LANE_COLOR_BGR
    return seg_bgr


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="SegFormer → semantic seg map")
    p.add_argument("--input_dir",  default=os.path.join(script_dir, "..", "dataset", "CAM_F0"))
    p.add_argument("--output_dir", default=os.path.join(script_dir, "..", "dataset", "CAM_F0_seg_segformer"))
    p.add_argument("--segformer_model",
                   default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading SegFormer...")
    processor, model = build_segformer(args.segformer_model, device)

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

        label_map = run_segformer(image_rgb, processor, model, device)
        seg       = label_to_color(label_map)
        seg       = overlay_lanes(image_rgb, label_map, seg)

        cv2.imwrite(os.path.join(args.output_dir,
                                 os.path.splitext(fname)[0] + ".png"), seg)

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {fname}")

    print(f"\nDone → {args.output_dir}")


if __name__ == "__main__":
    main()
