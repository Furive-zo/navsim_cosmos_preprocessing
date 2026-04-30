"""
RAM++ → GroundingDINO → SAM2 → semantic seg map

흐름:
  RAM++ : 이미지에서 객체 태그 자동 감지
  GroundingDINO : 태그 → bounding box
  SAM2 : box → 정밀 pixel mask
  → 태그별 색상으로 colored seg map 저장
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image as PILImage


# Cityscapes 기반 색상 (seg_generate.py 와 공유)
CITYSCAPES_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)

# 날씨/장면 관련 라벨만 통과 (보행자·소형 장애물 제외)
WEATHER_LABELS = {
    "vehicle", "car", "truck", "bus", "van", "motorcycle", "taxi", "suv",
    "road", "street", "pavement", "sidewalk",
    "sky", "cloud", "fog",
    "building", "house", "skyscraper", "wall", "fence",
    "vegetation", "tree", "grass", "bush",
    "traffic light", "traffic sign", "pole",
    "bridge", "tunnel", "highway",
}

def _idx_to_bgr(idx: int) -> list:
    """인스턴스 인덱스 → 고유 BGR 색상 (같은 라벨도 인스턴스마다 다른 색)."""
    h = int(idx * 137) % 180   # golden-angle hue spacing
    hsv = np.uint8([[[h, 220, 200]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_ram(checkpoint, device):
    from ram import get_transform, inference_ram
    from ram.models import ram_plus
    checkpoint = os.path.abspath(checkpoint)
    model = ram_plus(pretrained=checkpoint, image_size=384, vit="swin_l")
    model.eval()
    model.to(device)
    return model, get_transform(image_size=384), inference_ram


def build_grounding_dino(model_id, device):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def build_sam2(config, checkpoint, device):
    from sam2.build_sam import build_sam2 as _build
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(_build(config, checkpoint, device=device))


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

def detect_labels_ram(pil_image, ram_model, ram_transform, ram_infer, device):
    tensor = ram_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        english_tags, _ = ram_infer(tensor, ram_model)
    raw = [t.strip() for t in english_tags.split("|") if t.strip()]
    return [l for l in raw if l.lower() in WEATHER_LABELS]


def detect_boxes(pil_image, labels, gd_processor, gd_model, box_threshold, device):
    text = " . ".join(labels) + " ."
    inputs = gd_processor(images=pil_image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)
    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        target_sizes=[pil_image.size[::-1]],
    )[0]
    return results["boxes"].cpu().numpy(), results["labels"]


def build_seg_map(image_rgb, boxes, det_labels, sam_predictor):
    h, w = image_rgb.shape[:2]
    seg = np.zeros((h, w, 3), dtype=np.uint8)
    if len(boxes) == 0:
        return seg
    sam_predictor.set_image(image_rgb)
    for idx, (box, _) in enumerate(zip(boxes, det_labels)):
        masks, _, _ = sam_predictor.predict(box=box[None], multimask_output=False)
        seg[masks[0].astype(bool)] = _idx_to_bgr(idx)
    return seg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="RAM++ → GroundingDINO → SAM2 seg")
    p.add_argument("--input_dir",  default=os.path.join(script_dir, "..", "dataset", "CAM_F0"))
    p.add_argument("--output_dir", default=os.path.join(script_dir, "..", "dataset", "CAM_F0_seg_ram"))
    p.add_argument("--sam2_checkpoint", required=True)
    p.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--ram_checkpoint", default=None,
                   help="RAM++ .pth (미지정 시 --labels 필수)")
    p.add_argument("--grounding_model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--labels", default=None,
                   help="수동 라벨 지정 e.g. 'car,road,sky' (RAM++ 대체)")
    p.add_argument("--box_threshold", type=float, default=0.3)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.labels is None and args.ram_checkpoint is None:
        raise ValueError("--ram_checkpoint 또는 --labels 중 하나 필요")

    print("Loading SAM2...")
    sam_predictor = build_sam2(args.sam2_config, args.sam2_checkpoint, device)

    print("Loading GroundingDINO...")
    gd_processor, gd_model = build_grounding_dino(args.grounding_model, device)

    ram_model = ram_transform = ram_infer = None
    if args.labels is None:
        print("Loading RAM++...")
        ram_model, ram_transform, ram_infer = build_ram(args.ram_checkpoint, device)

    manual_labels = [l.strip() for l in args.labels.split(",")] if args.labels else None

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
        pil_image = PILImage.fromarray(image_rgb)

        labels = manual_labels or detect_labels_ram(
            pil_image, ram_model, ram_transform, ram_infer, device
        )

        if labels:
            boxes, det_labels = detect_boxes(
                pil_image, labels, gd_processor, gd_model, args.box_threshold, device
            )
            seg = build_seg_map(image_rgb, boxes, det_labels, sam_predictor)
        else:
            h, w = image_rgb.shape[:2]
            seg = np.zeros((h, w, 3), dtype=np.uint8)

        out_name = os.path.splitext(fname)[0] + ".png"
        cv2.imwrite(os.path.join(args.output_dir, out_name), seg)

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            tag_str = ", ".join(labels[:5]) + ("..." if len(labels) > 5 else "")
            print(f"  [{i+1}/{len(files)}] {fname}  [{tag_str}]")

    print(f"\nDone → {args.output_dir}")


if __name__ == "__main__":
    main()
