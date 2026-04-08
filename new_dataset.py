"""
new_dataset.py
==============
New_dataset
  - 입력 이미지(input)와 GT 이미지(got)를 쌍으로 로드
  - 학습 시 HorizontalFlip / VerticalFlip 동일하게 적용
  - torchvision.transforms 기반 PyTorch 순수 구현

디렉토리 구조 예시:
  input_dir/
    0001.png
    0002.png
    ...
  got_dir/
    0001.png
    0002.png
    ...
  → 파일명이 일치하는 쌍으로 매칭
"""

import random
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class New_dataset(Dataset):
    """
    입력 이미지와 GT(got) 이미지 쌍을 로드하는 Dataset

    Parameters
    ----------
    input_dir : 입력 이미지 폴더 경로
    got_dir   : GT 이미지 폴더 경로
    augment   : True  → HorizontalFlip + VerticalFlip 적용 (학습용)
                False → 변환 없음 (검증/테스트용)
    img_size  : 리사이즈할 이미지 크기 (None이면 원본 크기 유지)
    exts      : 허용 이미지 확장자
    """

    EXTS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}

    def __init__(self,
                 input_dir : str,
                 got_dir   : str,
                 augment   : bool = True,
                 img_size  : int  = None):

        self.input_dir = Path(input_dir)
        self.got_dir   = Path(got_dir)
        self.augment   = augment
        self.img_size  = img_size

        # 입력 이미지 목록 수집
        input_files = sorted([
            f for f in self.input_dir.iterdir()
            if f.suffix.lower() in self.EXTS
        ])

        if len(input_files) == 0:
            raise FileNotFoundError(f"입력 이미지 없음: {self.input_dir}")

        # got_dir에 동일 파일명 존재 확인 후 쌍 구성
        self.pairs = []
        for inp in input_files:
            got_path = self._find_pair(inp)
            if got_path is not None:
                self.pairs.append((inp, got_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"매칭되는 (input, got) 쌍 없음\n"
                f"  input_dir : {self.input_dir}\n"
                f"  got_dir   : {self.got_dir}"
            )

        print(f"  {'Train' if augment else 'Val'} 이미지 쌍: "
              f"{len(self.pairs):,}개  "
              f"{'(HFlip + VFlip 적용)' if augment else '(augmentation 없음)'}")

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _find_pair(self, inp: Path):
        """입력 파일명과 동일한 파일을 got_dir에서 탐색 (확장자 무관)"""
        # 1) 완전 동일한 파일명
        exact = self.got_dir / inp.name
        if exact.exists():
            return exact

        # 2) 확장자만 다른 경우
        for ext in self.EXTS:
            candidate = self.got_dir / (inp.stem + ext)
            if candidate.exists():
                return candidate

        return None

    def _apply_augment(self, inp_img: Image.Image,
                       got_img: Image.Image):
        """HorizontalFlip / VerticalFlip 을 두 이미지에 동일하게 적용"""
        if random.random() < 0.5:
            inp_img = TF.hflip(inp_img)
            got_img = TF.hflip(got_img)

        if random.random() < 0.5:
            inp_img = TF.vflip(inp_img)
            got_img = TF.vflip(got_img)

        return inp_img, got_img

    # ── Dataset 인터페이스 ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        inp_path, got_path = self.pairs[idx]

        inp_img = Image.open(inp_path).convert('RGB')
        got_img = Image.open(got_path).convert('RGB')

        # 리사이즈 (선택)
        if self.img_size is not None:
            inp_img = TF.resize(inp_img, [self.img_size, self.img_size])
            got_img = TF.resize(got_img, [self.img_size, self.img_size])

        # Augmentation (학습 시만)
        if self.augment:
            inp_img, got_img = self._apply_augment(inp_img, got_img)

        # PIL → Tensor [C, H, W], 값 범위 [0, 1]
        inp_t = TF.to_tensor(inp_img)
        got_t = TF.to_tensor(got_img)

        return {
            'id'   : inp_path.stem,
            'input': inp_t,   # [3, H, W] float32
            'got'  : got_t,   # [3, H, W] float32
        }
