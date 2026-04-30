# NAVSIM Cosmos Preprocessing

NAVSIM 데이터셋 카메라 이미지로부터 **Edge / Segmentation / Depth** 맵을 생성하는 전처리 파이프라인입니다.  
Cosmos world model 입력용 control map 생성을 목적으로 합니다.

## 생성 결과물

| 종류 | 스크립트 | 출력 폴더 | 모델 |
|------|----------|-----------|------|
| Edge map | `edge_generate.py` | `CAM_F0_edge` | OpenCV Canny |
| Seg (SegFormer) | `seg_generate.py` | `CAM_F0_seg_segformer` | SegFormer-B2 Cityscapes |
| Seg (RAM++ pipeline) | `seg_generate_ram.py` | `CAM_F0_seg_ram` | RAM++ → GroundingDINO → SAM2 |
| Seg (pkl GT boxes) | `seg_generate_from_pkl.py` | `CAM_F0_seg_pkl` | SegFormer + SAM2 (3D→2D 투영) |
| Depth map | `depth_generate.py` | `CAM_F0_depth` | Depth Anything V2 |

## 디렉토리 구조

```
cosmos/
├── scripts/
│   ├── run_generate.sh          # 통합 실행 스크립트
│   ├── edge_generate.py
│   ├── seg_generate.py
│   ├── seg_generate_ram.py
│   ├── seg_generate_from_pkl.py
│   └── depth_generate.py
├── checkpoints/                 # 모델 가중치 (git 제외)
│   ├── sam2.1_hiera_large.pt
│   └── ram_plus_swin_large_14m.pth
├── dataset/                     # 데이터셋 (git 제외)
│   ├── CAM_F0/                  # 원본 입력 이미지
│   ├── CAM_F0_edge/
│   ├── CAM_F0_seg_segformer/
│   ├── CAM_F0_seg_ram/
│   ├── CAM_F0_seg_pkl/
│   └── CAM_F0_depth/
└── example/                     # 샘플 영상
```

## 환경 설정

```bash
pip install opencv-python torch transformers pillow numpy
# SAM2
pip install git+https://github.com/facebookresearch/sam2.git
# RAM++ (seg_generate_ram.py 사용 시)
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

## 가중치 다운로드

```bash
mkdir -p checkpoints

# SAM2.1 Hiera Large
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# RAM++ Swin-Large 14M (seg_generate_ram.py 사용 시)
wget -P checkpoints https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth
```

HuggingFace 모델 (SegFormer, GroundingDINO, Depth Anything V2)은 실행 시 자동 다운로드됩니다.

## 실행 방법

### 통합 스크립트

`scripts/run_generate.sh` 상단의 플래그를 `1`로 변경 후 실행:

```bash
# run_generate.sh 내 플래그 수정
RUN_EDGE=1
RUN_SEG_SEGFORMER=1
RUN_DEPTH=1
# ...

bash scripts/run_generate.sh
```

### 개별 스크립트

**Edge**
```bash
python scripts/edge_generate.py
# 기본값: dataset/CAM_F0 → dataset/CAM_F0_edge
```

**Segmentation (SegFormer)**
```bash
python scripts/seg_generate.py \
    --input_dir  dataset/CAM_F0 \
    --output_dir dataset/CAM_F0_seg_segformer
```

**Segmentation (RAM++ → GroundingDINO → SAM2)**
```bash
python scripts/seg_generate_ram.py \
    --input_dir       dataset/CAM_F0 \
    --output_dir      dataset/CAM_F0_seg_ram \
    --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
    --ram_checkpoint  checkpoints/ram_plus_swin_large_14m.pth
# 라벨 수동 지정 시: --labels "car,road,sky,building"
```

**Segmentation (NAVSIM pkl GT boxes + SAM2)**
```bash
python scripts/seg_generate_from_pkl.py \
    --pkl_dir         dataset \
    --img_root        dataset/CAM_F0 \
    --output_dir      dataset/CAM_F0_seg_pkl \
    --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt
```

**Depth**
```bash
python scripts/depth_generate.py \
    --input_dir  dataset/CAM_F0 \
    --output_dir dataset/CAM_F0_depth \
    --model      depth-anything/Depth-Anything-V2-Large-hf
```

## Segmentation 파이프라인 비교

| 방식 | 특징 |
|------|------|
| SegFormer | 빠름, Cityscapes 19클래스, 차선 오버레이 포함 |
| RAM++ pipeline | 오픈 어휘, 정밀 instance mask, 느림 (GPU 권장) |
| pkl GT boxes | NAVSIM 3D annotation 활용, 차량 mask 정밀도 최고 |
