"""
extract_patches.py
==================
원본 이미지에서 패치를 추출해서 .npz로 저장
학습 전 1회 실행

실행:
  python extract_patches.py

저장 구조:
  patches/
    train/
      {id}_p{nn}_y{y1}_x{x1}.npz
    val/
      {id}_p{nn}_y{y1}_x{x1}.npz
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_utils import scan_visia_dataset
from face_mask import FaceMaskExtractor
from patch_sampler import FaceAwarePatchSampler


# ── 설정 ──────────────────────────────────────────────────────────────────────
CFG = dict(
    main_folder    = '/path/to/main_folder',   # ← 수정
    output_dir     = './patches',
    mask_cache_dir = './cache/face_masks',
    max_subjects   = 1000,
    mode           = 'random',
    seed           = 42,
    patch_size     = 224,    # 14의 배수 (DINOv2 호환)
    n_patches      = 16,     # 인물당 패치 수
    face_ratio     = 0.9,    # 검은 배경 → 얼굴 위주
    min_face_cover = 0.5,    # 패치 내 최소 얼굴 비율
)


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def remove_gamma(img: np.ndarray) -> np.ndarray:
    """uint8 sRGB → float32 linear RGB (IEC 61966-2-1)"""
    img = img.astype(np.float32) / 255.0
    return np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4
    )


def load_original(path: str, to_linear: bool = False) -> np.ndarray:
    """리사이즈 없이 원본 해상도 로드"""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"파일 로드 실패: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return remove_gamma(img) if to_linear else img.astype(np.float32) / 255.0


def get_face_mask(sample     : dict,
                  cache_dir  : Path,
                  extractor  : FaceMaskExtractor) -> np.ndarray:
    """
    원본 해상도 face mask
    MediaPipe는 512로 축소해서 계산 → 원본 크기로 복원
    """
    cache_path = cache_dir / f"{sample['id']}_mask.npy"
    if cache_path.exists():
        return np.load(str(cache_path))

    rgb_u8 = cv2.imread(str(sample['rgb']))
    rgb_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2RGB)
    H, W   = rgb_u8.shape[:2]

    # MediaPipe: 512로 축소해서 계산 (속도)
    rgb_small  = cv2.resize(rgb_u8, (512, 512))
    mask_small = extractor.get_mask(rgb_small)

    # 원본 해상도로 복원
    mask = cv2.resize(
        mask_small, (W, H), interpolation=cv2.INTER_NEAREST
    ).astype(np.float32)

    np.save(str(cache_path), mask)
    return mask


# ── 패치 추출 ─────────────────────────────────────────────────────────────────
def extract_and_save(dataset_info : list,
                     split        : str,
                     output_dir   : Path,
                     mask_cache   : Path,
                     cfg          : dict) -> int:
    """
    dataset_info의 모든 인물에서 패치 추출 후 .npz 저장

    Parameters
    ----------
    split : 'train' or 'val'

    Returns
    -------
    저장된 총 패치 수
    """
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    extractor = FaceMaskExtractor()
    sampler   = FaceAwarePatchSampler(
        patch_size     = cfg['patch_size'],
        n_patches      = cfg['n_patches'],
        face_ratio     = cfg['face_ratio'],
        min_face_cover = cfg['min_face_cover'],
    )
    p             = cfg['patch_size']
    total_patches = 0
    skipped       = 0

    for i, sample in enumerate(dataset_info):
        pid = sample['id']

        # 이미 추출된 인물 스킵
        existing = list(split_dir.glob(f'{pid}_p*.npz'))
        if len(existing) >= cfg['n_patches']:
            total_patches += len(existing)
            continue

        try:
            # 원본 해상도 로드 (리사이즈 없음)
            rgb   = load_original(sample['rgb'],   to_linear=True)
            brown = load_original(sample['brown'], to_linear=False)
            red   = load_original(sample['red'],   to_linear=False)
            mask  = get_face_mask(sample, mask_cache, extractor)

            H, W = rgb.shape[:2]

            if H < p or W < p:
                print(f"  ⚠️  {pid} 크기({H}×{W}) < patch({p}) 스킵")
                skipped += 1
                continue

            # GT grayscale 변환
            mel_map = brown.mean(axis=2)   # [H, W]
            hem_map = red.mean(axis=2)     # [H, W]

            # 패치 위치 샘플링
            positions = sampler.sample(mask)

            # 패치 추출 및 .npz 저장
            saved = 0
            for patch_idx, (y1, x1) in enumerate(positions):
                y2, x2  = y1 + p, x1 + p

                fmask_p = mask[y1:y2, x1:x2]

                # 얼굴 비율 체크
                if fmask_p.mean() < cfg['min_face_cover']:
                    continue

                rgb_p = rgb  [y1:y2, x1:x2]       # [p, p, 3]
                mel_p = mel_map[y1:y2, x1:x2]     # [p, p]
                hem_p = hem_map[y1:y2, x1:x2]     # [p, p]

                # [H, W, C] → [C, H, W]
                rgb_t   = rgb_p.transpose(2, 0, 1).astype(np.float32)
                mel_t   = mel_p[np.newaxis].astype(np.float32)
                hem_t   = hem_p[np.newaxis].astype(np.float32)
                fmask_t = fmask_p[np.newaxis].astype(np.float32)

                fname = f'{pid}_p{patch_idx:02d}_y{y1}_x{x1}.npz'
                np.savez_compressed(
                    str(split_dir / fname),
                    rgb   = rgb_t,
                    mel   = mel_t,
                    hem   = hem_t,
                    fmask = fmask_t,
                    meta  = np.array([y1, x1, H, W]),
                )
                saved         += 1
                total_patches += 1

        except Exception as e:
            print(f"  ❌ {pid} 오류: {e}")
            skipped += 1
            continue

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(dataset_info)}] "
                  f"누적 패치: {total_patches:,}")

    extractor.close()
    print(f"✅ {split}: {total_patches:,}개 저장  (스킵: {skipped}명)")
    return total_patches


# ── 실행 ──────────────────────────────────────────────────────────────────────
def main():
    output_dir = Path(CFG['output_dir'])
    mask_cache = Path(CFG['mask_cache_dir'])
    mask_cache.mkdir(parents=True, exist_ok=True)

    dataset_info = scan_visia_dataset(
        CFG['main_folder'],
        max_subjects = CFG['max_subjects'],
        mode         = CFG['mode'],
        seed         = CFG['seed'],
    )
    train_info, val_info = train_test_split(
        dataset_info, test_size=0.2, random_state=CFG['seed']
    )

    print(f"\nTrain: {len(train_info)}명 | Val: {len(val_info)}명")
    print(f"인물당 패치: {CFG['n_patches']}개  |  패치 크기: {CFG['patch_size']}×{CFG['patch_size']}")
    print(f"예상 총 패치: {len(dataset_info) * CFG['n_patches']:,}개\n")

    print("=== Train 패치 추출 ===")
    n_train = extract_and_save(train_info, 'train', output_dir, mask_cache, CFG)

    print("\n=== Val 패치 추출 ===")
    n_val = extract_and_save(val_info, 'val', output_dir, mask_cache, CFG)

    total_size = sum(f.stat().st_size for f in output_dir.rglob('*.npz'))
    print(f"\n=== 완료 ===")
    print(f"  Train: {n_train:,}개")
    print(f"  Val  : {n_val:,}개")
    print(f"  용량 : {total_size/1e9:.2f} GB")
    print(f"  경로 : {output_dir.resolve()}")


if __name__ == '__main__':
    main()
