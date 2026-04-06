"""
ambient_aug.py
==============
주변광(Ambient Light) 변화 시뮬레이션 Augmentation

학습 시 원본 패치와 조명변형 패치를 쌍으로 생성
→ DINOv2 feature consistency loss에 사용

물리적 근거:
  실제 피부 촬영 환경에서 발생하는 조명 변화:
  - 형광등 / 백열등 / LED 색온도 차이
  - 조명 세기 (밝기) 차이
  - 광원 색상 (RGB shift)
  - 카메라 노출 변화
"""

import numpy as np
import albumentations as A


def get_ambient_transform() -> A.Compose:
    """
    주변광 변화 시뮬레이션 Augmentation

    linear RGB에서 적용
    강도는 실제 피부 촬영 환경 변화 범위로 설정

    Returns
    -------
    albumentations Compose (image only)
    """
    return A.Compose([

        # ── 조명 세기 변화 (밝기 / 대비) ──────────────────────────────────
        # 형광등 vs 자연광 세기 차이 시뮬레이션
        A.RandomBrightnessContrast(
            brightness_limit = 0.3,    # ±30% 밝기 변화
            contrast_limit   = 0.2,    # ±20% 대비 변화
            p                = 0.8
        ),

        # ── 색온도 변화 (형광등 vs 백열등 vs LED) ────────────────────────
        A.ColorJitter(
            hue        = 0.0,    # 색상 변화 없음 (피부 색조 보존)
            saturation = 0.15,   # 채도 약간 변화
            brightness = 0.0,    # brightness는 위에서 처리
            contrast   = 0.0,
            p          = 0.7
        ),

        # ── 광원 색상 변화 (RGB channel shift) ───────────────────────────
        # 황색 조명(R↑, B↓) vs 청색 조명(B↑, R↓)
        A.RGBShift(
            r_shift_limit = 20,
            g_shift_limit = 15,
            b_shift_limit = 20,
            p             = 0.6
        ),

        # ── 카메라 노출 변화 (Gamma) ──────────────────────────────────────
        A.RandomGamma(
            gamma_limit = (70, 130),   # 0.7 ~ 1.3 gamma
            p           = 0.5
        ),

    ])


def apply_ambient_aug(transform : A.Compose,
                      rgb_batch : np.ndarray) -> np.ndarray:
    """
    배치 전체에 주변광 augmentation 적용

    Parameters
    ----------
    transform  : get_ambient_transform() 반환값
    rgb_batch  : [B, H, W, 3] float32 linear RGB

    Returns
    -------
    rgb_aug : [B, H, W, 3] float32  조명변형 이미지
    """
    aug_list = []
    for img in rgb_batch:
        # albumentations는 float32 [0,1] 지원
        out = transform(image=img)['image']
        aug_list.append(out)
    return np.stack(aug_list, axis=0)
