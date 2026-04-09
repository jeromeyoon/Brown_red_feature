# dino-UNet

기존 ChromophoreNet과 구분되는 DinoUNet 구현체

## 기존 ChromophoreNet과의 차이

| 항목 | ChromophoreNet (기존) | DinoUNet (신규) |
|---|---|---|
| Decoder | 외부 사전학습 .pth 필요 | 내장 UNetDecoder (처음부터 학습) |
| Skip Connection | Bridge에서 단순 형식 변환 | UpBlock에서 명시적 skip concat |
| 학습 전략 | freeze_dino 단일 옵션 | Phase 1 → Phase 2 자동 전환 |
| Decoder 채널 확인 | inspect_decoder_channels() 필요 | 불필요 |

## 파일 구조

```
dino-UNet/
  model.py    ← DinoUNet (FeatureAdapter + UNetDecoder)
  loss.py     ← ChromophoreLoss + AmbientInvariantLoss (기존과 동일)
  train.py    ← Phase 1/2 학습 스크립트
  README.md
```

공유 모듈 (상위 폴더 재사용):
- `../dataset.py`      → PatchDataset
- `../ambient_aug.py`  → 조명 augmentation
- `../patch_sampler.py`
- `../face_mask.py`

## 아키텍처

```
RGB [B, 3, H, W]
  ↓
DINOv2 ViT-B
  ↓ layer 3, 6, 9, 12 중간 레이어 피처
  ↓
FeatureAdapter (ViT 토큰 → 공간 피처맵)
  bottleneck [B, 512, H/32, W/32]
  skip3      [B, 256, H/16, W/16]
  skip2      [B, 128, H/8,  W/8 ]
  skip1      [B,  64, H/4,  W/4 ]
  ↓
UNetDecoder
  UpBlock × 3 (skip connection 포함)
  + ConvTranspose2d
  ↓
mel_pred [B, 1, H, W]
hem_pred [B, 1, H, W]
```

## 학습 전략

**Phase 1 (epoch 1 ~ 29)**
- DINOv2 완전 frozen
- FeatureAdapter + UNetDecoder만 학습
- lr = 1e-4

**Phase 2 (epoch 30 ~ 80)**
- DINOv2 마지막 4블록 unfreeze
- end-to-end fine-tuning
- DINOv2 lr = 1e-5 (별도 param group)

## 실행

```bash
cd dino-UNet

# 학습
python train.py

# 백그라운드 실행
nohup python train.py > train_dino_unet.log 2>&1 &
```

## CFG 수정 필요 항목

```python
CFG = dict(
    patch_dir   = '../patches',               # extract_patches.py 출력 폴더
    dinov2_path = '../pretrained/dinov2-base', # DINOv2 로컬 경로
    ...
)
```
