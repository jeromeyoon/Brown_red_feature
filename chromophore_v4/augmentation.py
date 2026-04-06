"""
augmentation.py
===============
학습 시 온라인 Augmentation

패치는 사전 추출 완료 상태
학습 시에만 flip 적용 (텍스처 보존 목적)

적용:
  - HorizontalFlip : 좌우 반전
  - VerticalFlip   : 상하 반전

미적용 (텍스처 손상 우려):
  - Rotate         : 회전 시 border padding → 텍스처 왜곡
  - RandomResizedCrop : 리사이즈 → 텍스처 손실
  - 강한 photometric  : linear RGB 물리 법칙 위반

Notes
-----
additional_targets:
  mel   → 'mask'  (RGB와 동일한 geometric 변환, nearest 보간)
  hem   → 'mask'
  fmask → 'mask'
"""

import albumentations as A


def get_train_transform() -> A.Compose:
    """
    학습용: HorizontalFlip + VerticalFlip
    rgb / mel / hem / fmask 동일 변환 적용
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], additional_targets={
        'mel'  : 'mask',
        'hem'  : 'mask',
        'fmask': 'mask',
    })


def get_val_transform() -> A.Compose:
    """검증/테스트용: 변환 없음"""
    return A.Compose(
        [],
        additional_targets={
            'mel'  : 'mask',
            'hem'  : 'mask',
            'fmask': 'mask',
        }
    )


def apply_transform(transform : A.Compose,
                    rgb       ,   # [H, W, 3] float32
                    mel       ,   # [H, W]    float32
                    hem       ,   # [H, W]    float32
                    fmask     ):  # [H, W]    float32
    """
    rgb / mel / hem / fmask 에 동일한 변환 적용

    Returns
    -------
    (rgb, mel, hem, fmask) 변환 적용된 numpy array
    """
    out = transform(
        image  = rgb,
        mel    = mel,
        hem    = hem,
        fmask  = fmask,
    )
    return out['image'], out['mel'], out['hem'], out['fmask']
