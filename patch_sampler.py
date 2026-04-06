import numpy as np


class FaceAwarePatchSampler:
    def __init__(self,
                 patch_size    : int   = 128,
                 n_patches     : int   = 32,
                 face_ratio    : float = 0.7,
                 min_face_cover: float = 0.5):
        """
        patch_size    : 패치 크기 (정사각형)
        n_patches     : 이미지 1장당 샘플링할 패치 수
        face_ratio    : 전체 패치 중 얼굴 영역 비율
        min_face_cover: 패치 내 얼굴 픽셀 최소 비율
        """
        self.patch_size     = patch_size
        self.n_patches      = n_patches
        self.face_ratio     = face_ratio
        self.min_face_cover = min_face_cover

    def _sample_positions(self, mask: np.ndarray, n: int,
                          prefer_face: bool) -> list:
        H, W  = mask.shape
        p     = self.patch_size
        positions = []
        max_try   = n * 50

        if prefer_face:
            prob_map = mask.copy()
        else:
            prob_map = (1.0 - mask)

        if prob_map.sum() < 1:
            prob_map = np.ones_like(mask)

        # 패치 중심 가능 범위로 마진 제거
        prob_map[:p//2, :]  = 0
        prob_map[-p//2:, :] = 0
        prob_map[:, :p//2]  = 0
        prob_map[:, -p//2:] = 0

        if prob_map.sum() < 1:
            prob_map = np.ones_like(mask)
            prob_map[:p//2, :]  = 0
            prob_map[-p//2:, :] = 0
            prob_map[:, :p//2]  = 0
            prob_map[:, -p//2:] = 0

        flat_prob = prob_map.flatten()
        flat_prob = flat_prob / flat_prob.sum()

        tried = 0
        while len(positions) < n and tried < max_try:
            tried += 1
            idx    = np.random.choice(len(flat_prob), p=flat_prob)
            cy, cx = divmod(idx, W)

            y1, y2 = cy - p // 2, cy + p // 2
            x1, x2 = cx - p // 2, cx + p // 2

            if y1 < 0 or y2 > H or x1 < 0 or x2 > W:
                continue

            patch_mask = mask[y1:y2, x1:x2]
            face_cover = patch_mask.mean()

            if prefer_face and face_cover < self.min_face_cover:
                continue

            positions.append((y1, x1))

        return positions

    def sample(self, mask: np.ndarray) -> list:
        """
        mask   : [H, W] float32 face mask
        returns: [(y1, x1), ...] 패치 좌표 목록
        """
        n_face = int(self.n_patches * self.face_ratio)
        n_bg   = self.n_patches - n_face

        face_pos = self._sample_positions(mask, n_face, prefer_face=True)
        bg_pos   = self._sample_positions(mask, n_bg,   prefer_face=False)

        all_pos = face_pos + bg_pos
        np.random.shuffle(all_pos)
        return all_pos
