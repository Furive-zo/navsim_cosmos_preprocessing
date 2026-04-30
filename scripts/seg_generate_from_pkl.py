"""
NAVSIM pkl → SegFormer + SAM2 seg mask 생성

흐름:
  SegFormer (Cityscapes) → semantic base map (도로/건물/하늘/사람 등)
  pkl gt_boxes 3D → 2D 투영 → SAM2 → 정밀 차량 mask 덮어쓰기
"""

import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Cityscapes 팔레트 (seg_generate.py 와 동일)
# ---------------------------------------------------------------------------

CITYSCAPES_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)

LANE_COLOR_BGR   = (0, 255, 255)   # 차선
# 차량 계열 label ID (car/truck/bus/train/motorcycle/bicycle)
VEHICLE_IDS      = {13, 14, 15, 16, 17, 18}


def build_segformer(model_id, device):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    processor = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def run_segformer(image_rgb, processor, model, device):
    H, W  = image_rgb.shape[:2]
    inputs = processor(images=PILImage.fromarray(image_rgb), return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=(H, W), mode="bilinear", align_corners=False
    )
    return up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def base_seg_map(image_rgb, label_map):
    """SegFormer 결과 → BGR seg map + 차선 오버레이."""
    rgb = CITYSCAPES_PALETTE[label_map]
    seg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    road_mask = np.isin(label_map, [0, 1, 9])
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    white  = (hsv[:, :, 1] < 50)  & (hsv[:, :, 2] > 180)
    yellow = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35) & \
             (hsv[:, :, 1] > 80)  & (hsv[:, :, 2] > 100)
    seg[road_mask & (white | yellow)] = LANE_COLOR_BGR
    return seg


def vehicle_color_from_segformer(label_map):
    """SegFormer가 해당 이미지에서 검출한 차량 픽셀의 대표 색 반환."""
    vehicle_mask = np.isin(label_map, list(VEHICLE_IDS))
    if not vehicle_mask.any():
        return (0, 0, 142)
    ids, counts = np.unique(label_map[vehicle_mask], return_counts=True)
    dominant_id = ids[counts.argmax()]
    rgb = CITYSCAPES_PALETTE[dominant_id].tolist()
    return (rgb[2], rgb[1], rgb[0])


# ---------------------------------------------------------------------------
# 3D bbox → 2D projection  (from seg_generate_from_pkl reference)
# ---------------------------------------------------------------------------

def box_to_corners(box: np.ndarray) -> np.ndarray:
    """gt_box [x,y,z,w,l,h,yaw] → 8 corners [8,3] in lidar/ego frame."""
    x, y, z, w, l, h, yaw = box
    dx, dy, dz = w / 2, l / 2, h / 2
    local = np.array([
        [ dx,  dy, -dz], [ dx, -dy, -dz],
        [-dx, -dy, -dz], [-dx,  dy, -dz],
        [ dx,  dy,  dz], [ dx, -dy,  dz],
        [-dx, -dy,  dz], [-dx,  dy,  dz],
    ])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_y, -sin_y, 0],
                  [sin_y,  cos_y, 0],
                  [0,      0,     1]])
    return local @ R.T + np.array([x, y, z])


def lidar_to_cam(points: np.ndarray, cam: dict) -> np.ndarray:
    R = np.array(cam["sensor2lidar_rotation"])
    t = np.array(cam["sensor2lidar_translation"])
    R_inv = R.T
    t_inv = -R_inv @ t
    return points @ R_inv.T + t_inv


def project_to_image(points_cam: np.ndarray, cam: dict):
    """Returns (pixels [N,2], valid_mask [N])."""
    K    = np.array(cam["cam_intrinsic"],  dtype=np.float64)
    dist = np.array(cam["distortion"],     dtype=np.float64)

    valid = points_cam[:, 2] > 0.1
    if not valid.any():
        return np.zeros((len(points_cam), 2)), valid

    pts_2d, _ = cv2.projectPoints(
        points_cam[valid].reshape(-1, 1, 3),
        np.zeros(3), np.zeros(3), K, dist,
    )
    result = np.zeros((len(points_cam), 2))
    result[valid] = pts_2d.reshape(-1, 2)
    return result, valid


def corners_to_2d_box(corners_lidar, cam, img_w, img_h):
    """
    3D box corners → 2D axis-aligned bounding box [x1,y1,x2,y2].
    Returns None if fewer than 4 corners project into the image.
    """
    corners_cam        = lidar_to_cam(corners_lidar, cam)
    pixels, valid      = project_to_image(corners_cam, cam)

    if valid.sum() < 4:
        return None

    pts = pixels[valid].astype(np.float32)
    in_bounds = (
        (pts[:, 0] >= 0) & (pts[:, 0] < img_w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < img_h)
    )
    if in_bounds.sum() < 3:
        return None

    pts = pts[in_bounds]
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()

    # 박스가 너무 작으면 스킵 (2px 이하)
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ---------------------------------------------------------------------------
# SAM2
# ---------------------------------------------------------------------------

def build_sam2(config: str, checkpoint: str, device: str):
    from sam2.build_sam import build_sam2 as _build
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(_build(config, checkpoint, device=device))


# ---------------------------------------------------------------------------
# Core: one frame → colored seg map
# ---------------------------------------------------------------------------

def frame_to_seg(
    frame: dict,
    cam_name: str,
    img_root: Path,
    sam_predictor,
    seg_processor,
    seg_model,
    device: str,
    target_classes: list | None = None,
) -> np.ndarray | None:
    """
    1) SegFormer → Cityscapes semantic base map (도로/건물/하늘/차선 등)
    2) SAM2 + pkl 3D boxes → 정밀 차량 mask 덮어쓰기
    Returns BGR colored seg map (H, W, 3), or None if image not found.
    """
    cam  = frame["cams"][cam_name]
    anns = frame["anns"]

    gt_boxes = np.array(anns["gt_boxes"])  # [N, 7]
    gt_names = np.array(anns["gt_names"])  # [N]

    if target_classes:
        keep     = np.isin(gt_names, target_classes)
        gt_boxes = gt_boxes[keep]
        gt_names = gt_names[keep]

    img_path = img_root / cam["data_path"]
    if not img_path.exists():
        img_path = img_root / os.path.basename(cam["data_path"])
    if not img_path.exists():
        return None

    img_bgr  = cv2.imread(str(img_path))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W     = img_bgr.shape[:2]

    # --- Step 1: SegFormer base map --------------------------------------
    label_map = run_segformer(img_rgb, seg_processor, seg_model, device)
    seg       = base_seg_map(img_rgb, label_map)

    if len(gt_boxes) == 0:
        return seg

    # --- Step 2: collect 2D boxes from pkl -------------------------------
    boxes_2d, names_2d = [], []
    for box, name in zip(gt_boxes, gt_names):
        b2d = corners_to_2d_box(box_to_corners(box), cam, W, H)
        if b2d is not None:
            boxes_2d.append(b2d)
            names_2d.append(name)

    if not boxes_2d:
        return seg

    # --- Step 3: SAM2 → precise masks, override SegFormer ----------------
    vehicle_color = vehicle_color_from_segformer(label_map)
    sam_predictor.set_image(img_rgb)
    masks_all, _, _ = sam_predictor.predict(
        box=np.stack(boxes_2d),
        multimask_output=False,
    )
    for mask in masks_all:
        seg[mask.squeeze().astype(bool)] = vehicle_color

    return seg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="NAVSIM pkl + SAM2 → colored seg masks"
    )
    p.add_argument(
        "--pkl_dir",
        required=True,
        help="NAVSIM log pkl 파일들이 있는 디렉터리",
    )
    p.add_argument(
        "--img_root",
        required=True,
        help="sensor_blobs 루트 (cam['data_path'] 기준)",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(script_dir, "..", "dataset", "CAM_F0_seg_pkl"),
    )
    p.add_argument(
        "--cam_name",
        default="CAM_F0",
        help="처리할 카메라 이름",
    )
    p.add_argument(
        "--sam2_checkpoint",
        required=True,
        help="SAM2 체크포인트 .pt",
    )
    p.add_argument(
        "--sam2_config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    p.add_argument(
        "--target_classes",
        default="vehicle",
        help="SAM2로 정밀 처리할 클래스 (쉼표 구분). 기본: vehicle",
    )
    p.add_argument(
        "--segformer_model",
        default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    )
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    img_root = Path(args.img_root)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_classes = [c.strip() for c in args.target_classes.split(",")]

    print("Loading SegFormer...")
    seg_processor, seg_model = build_segformer(args.segformer_model, device)

    print("Loading SAM2...")
    sam_predictor = build_sam2(args.sam2_config, args.sam2_checkpoint, device)

    pkl_files = sorted(Path(args.pkl_dir).glob("*.pkl"))
    assert pkl_files, f"No pkl files found in {args.pkl_dir}"
    print(f"Found {len(pkl_files)} pkl files")

    total = saved = 0
    for pkl_path in pkl_files:
        scene = pickle.load(open(pkl_path, "rb"))
        for frame in scene:
            total += 1
            # 원본 이미지 파일명과 일치하도록 data_path basename 사용
            cam_data_path = frame["cams"][args.cam_name].get("data_path", "")
            stem = os.path.splitext(os.path.basename(cam_data_path))[0]
            if not stem:
                stem = frame.get("token", f"frame_{total:06d}")

            seg = frame_to_seg(
                frame, args.cam_name, img_root,
                sam_predictor, seg_processor, seg_model, device,
                target_classes,
            )
            if seg is None:
                continue

            out_path = out_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), seg)
            saved += 1

            if saved % 10 == 0:
                print(f"  [{saved}] {stem}")

    print(f"\nDone: {saved}/{total} frames → {out_dir}")


if __name__ == "__main__":
    main()
