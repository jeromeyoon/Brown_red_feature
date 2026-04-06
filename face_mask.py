import cv2
import numpy as np
import mediapipe as mp


class FaceMaskExtractor:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_mask(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """
        rgb_uint8: [H, W, 3] uint8
        returns:   [H, W] binary mask (float32, 0 or 1)
        얼굴 미검출 시 전체 1 반환
        """
        H, W = rgb_uint8.shape[:2]
        result = self.face_mesh.process(rgb_uint8)

        if not result.multi_face_landmarks:
            print("  ⚠️  얼굴 미검출 → 전체 영역 사용")
            return np.ones((H, W), dtype=np.float32)

        landmarks = result.multi_face_landmarks[0].landmark
        pts = np.array([
            [int(lm.x * W), int(lm.y * H)]
            for lm in landmarks
        ], dtype=np.int32)

        mask = np.zeros((H, W), dtype=np.uint8)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask.astype(np.float32)

    def close(self):
        self.face_mesh.close()


def precompute_masks(dataset_info: list, img_size: int = 512,
                     cache_dir: str = './cache/face_masks'):
    """
    학습 전 모든 face mask를 미리 계산해서 .npy로 저장
    → DataLoader worker에서 MediaPipe 충돌 방지
    """
    from pathlib import Path
    extractor  = FaceMaskExtractor()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for i, d in enumerate(dataset_info):
        pid       = d['id']
        save_path = cache_path / f'{pid}_mask.npy'

        if save_path.exists():
            continue

        img = cv2.imread(str(d['rgb']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        mask = extractor.get_mask(img)
        np.save(str(save_path), mask)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(dataset_info)} 마스크 생성 완료")

    extractor.close()
    print("✅ 전체 마스크 캐시 완료")
