# Chromophore v4
## DINOv2 Ambient Invariant Feature + Melanin/Hemoglobin 분리

---

## 핵심 설계

```
[학습 목표]
DINOv2가 주변광에 강인한 feature를 학습
  → Decoder 연결 시 조명이 달라져도 동일한 mel/hem 출력 보장

[Loss 구성]
total = loss_chroma + w_feat * loss_feat

  loss_chroma : mel/hem 분리 (Supervised + SSIM + Recon + Ortho + Smooth)
  loss_feat   : 주변광 불변 feature consistency
                원본 feature ↔ 조명변형 feature → 동일해야 함
```

---

## 파일 구성

```
chromophore_v4/
├── train.py            # 학습 메인 (두 Loss 통합)
├── model.py            # ChromophoreNet (return_feat 지원)
├── loss.py             # ChromophoreLoss + AmbientInvariantLoss
├── ambient_aug.py      # 주변광 변화 시뮬레이션 Augmentation
├── dataset.py          # PatchDataset (HFlip + VFlip)
├── augmentation.py     # Flip augmentation
├── extract_patches.py  # 패치 추출 (1회 실행)
├── face_mask.py        # MediaPipe face mask
├── patch_sampler.py    # Face-aware patch 샘플링
├── data_utils.py       # VISIA 폴더 스캔
└── requirements.txt
```

---

## 데이터 흐름

```
DataLoader
  batch['rgb']  [B,3,H,W] linear
       ↓                    ↓
  ChromophoreNet         ambient_aug
  (return_feat=True)    (조명변형)
       ↓                    ↓
  mel_pred              ChromophoreNet
  hem_pred              (return_feat=True)
  cls_orig                   ↓
  patch_orig            cls_aug
                        patch_aug
       ↓                    ↓
  ChromophoreLoss    AmbientInvariantLoss
  (mel/hem 분리)     (feature consistency)
       ↓                    ↓
       └──── total loss ────┘
```

---

## Feature Consistency Loss 방법 비교

| 방법 | 특징 | 추천 상황 |
|------|------|---------|
| **cosine** | 방향만 일치 (크기 무관) | 기본 추천 ⭐ |
| mse | 방향 + 크기 모두 일치 | 강한 제약 필요 시 |
| **infonce** | Contrastive (같은 쌍 ↔ 다른 쌍) | 배치 크기 클 때 ⭐ |

---

## 실행 순서

### 1. 환경 설치
```bash
pip install -r requirements.txt
```

### 2. 패치 추출 (1회)
```bash
# extract_patches.py CFG 수정 후
python extract_patches.py
```

### 3. train.py 설정
```python
CFG = dict(
    dinov2_path    = './pretrained/dinov2-base',
    decoder_ckpt   = './pretrained/decoder.pth',
    dec_chs        = [512, 256, 128, 64],   # Decoder 채널 확인 후 수정
    feat_loss_type = 'cosine',   # 'cosine' | 'mse' | 'infonce'
    w_feat         = 0.5,        # feature consistency loss 가중치
    use_patch_feat = False,      # patch token도 사용 시 True
    ...
)
```

### 4. Decoder 클래스 연결
```python
# train.py 수정
from your_decoder import YourDecoder
decoder = load_pretrained_decoder(
    CFG['decoder_ckpt'], YourDecoder, CFG['decoder_prefix']
)
```

### 5. 학습
```bash
python train.py
nohup python train.py > train.log 2>&1 &
```

---

## Loss 가중치 스케줄

| Phase | Epoch | Supervised | SSIM | Recon | Ortho | Smooth | Feat |
|-------|-------|-----------|------|-------|-------|--------|------|
| 1 | 0~30% | 1.0 | 0.0 | 0.5 | 0.0 | 0.0 | w_feat |
| 2 | 30~70% | 1.0 | 0.3 | 0.5 | 0.2 | 0.1 | w_feat |
| 3 | 70~100% | 1.0 | 0.5 | 0.5 | 0.3 | 0.1 | w_feat |

feature consistency loss 가중치(w_feat)는 전 구간 고정
