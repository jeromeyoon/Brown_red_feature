"""
dataset.py
==========
PatchDataset
  - 사전 추출된 .npz 패치 파일 로드
  - 학습 시 HorizontalFlip / VerticalFlip 온라인 적용
  - rgb / mel_gt / hem_gt / face_mask 반환

.npz 파일 구조:
  rgb   : [3, H, W] float32 linear
  mel   : [1, H, W] float32
  hem   : [1, H, W] float32
  fmask : [1, H, W] float32
  meta  : [y1, x1, orig_H, orig_W]
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from augmentation import get_train_transform, get_val_transform, apply_transform


class PatchDataset(Dataset):
    """
    사전 추출된 .npz 패치 기반 Dataset

    Parameters
    ----------
    patch_dir : 패치 .npz 파일 폴더 (train/ or val/)
    augment   : True  → HorizontalFlip + VerticalFlip 적용 (학습용)
                False → 변환 없음 (검증/테스트용)
    """

    def __init__(self,
                 patch_dir : str,
                 augment   : bool = True):

        self.patch_dir = Path(patch_dir)
        self.augment   = augment
        self.files     = sorted(self.patch_dir.glob('*.npz'))

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"패치 파일 없음: {self.patch_dir}\n"
                f"extract_patches.py 먼저 실행하세요."
            )

        # 학습: flip augmentation / 검증: 변환 없음
        self.transform = (get_train_transform() if augment
                          else get_val_transform())

        print(f"  {'Train' if augment else 'Val'} 패치: "
              f"{len(self.files):,}개  "
              f"{'(HFlip + VFlip 적용)' if augment else '(augmentation 없음)'}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        # .npz 로드
        data  = np.load(str(self.files[idx]))

        # [C, H, W] → [H, W, C] or [H, W] (albumentations 입력 형식)
        rgb   = data['rgb'].transpose(1, 2, 0)   # [H, W, 3]
        mel   = data['mel'][0]                    # [H, W]
        hem   = data['hem'][0]                    # [H, W]
        fmask = data['fmask'][0]                  # [H, W]

        # ── Augmentation (학습 시만) ──────────────────────────────────
        # rgb / mel / hem / fmask 동일한 flip 변환 적용
        rgb, mel, hem, fmask = apply_transform(
            self.transform, rgb, mel, hem, fmask
        )

        # [H, W, C] → [C, H, W] Tensor
        rgb_t   = torch.from_numpy(
            rgb.transpose(2, 0, 1).copy()
        ).float()
        mel_t   = torch.from_numpy(mel[np.newaxis].copy()).float()
        hem_t   = torch.from_numpy(hem[np.newaxis].copy()).float()
        fmask_t = torch.from_numpy(fmask[np.newaxis].copy()).float()

        # 파일명에서 ID 파싱 (예: 0001_p00_y128_x256.npz)
        pid = self.files[idx].stem.split('_')[0]

        return {
            'id'       : pid,
            'rgb'      : rgb_t,     # [3, H, W] linear  ← ChromophoreNet 입력
            'mel_gt'   : mel_t,     # [1, H, W]          ← Loss GT
            'hem_gt'   : hem_t,     # [1, H, W]          ← Loss GT
            'face_mask': fmask_t,   # [1, H, W]          ← Loss masking
        }
