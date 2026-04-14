"""
loss.py
=======
ChromophoreLoss   : melanin / hemoglobin 분리 Loss
AmbientInvariantLoss : 주변광 불변 feature consistency Loss

전체 Loss:
  total = loss_chroma + w_feat * loss_feat

  loss_chroma:
    1. Supervised L1   : VISIA GT와 직접 비교
    2. SSIM            : 공간 구조 유사도
    3. Reconstruction  : Beer-Lambert 역산 RGB 비교
    4. Orthogonality   : mel/hem 독립성
    5. Smoothness (TV) : 공간 연속성

  loss_feat (AmbientInvariantLoss):
    원본 feature vs 조명변형 feature → 동일해야 함
    방법: Cosine | MSE | InfoNCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# SSIM
# ══════════════════════════════════════════════════════════════════════════════
class SSIMLoss(nn.Module):
    """Structural Similarity Loss  (1 - SSIM)"""

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('window', self._make_window(window_size))

    def _make_window(self, size: int) -> torch.Tensor:
        sigma  = size / 6.0
        coords = torch.arange(size).float() - size // 2
        g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g      = g / g.sum()
        w      = g.outer(g)
        return w.unsqueeze(0).unsqueeze(0)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        w, pad = self.window.to(pred), self.window_size // 2

        mu1    = F.conv2d(pred,      w, padding=pad)
        mu2    = F.conv2d(gt,        w, padding=pad)
        sigma1 = F.conv2d(pred*pred, w, padding=pad) - mu1 ** 2
        sigma2 = F.conv2d(gt  *gt,   w, padding=pad) - mu2 ** 2
        sigma12= F.conv2d(pred*gt,   w, padding=pad) - mu1 * mu2

        ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        return 1 - ssim_map.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Chromophore Loss
# ══════════════════════════════════════════════════════════════════════════════
class ChromophoreLoss(nn.Module):
    """
    Melanin / Hemoglobin 분리 Loss

    Parameters
    ----------
    w_supervised : Supervised L1 가중치
    w_ssim       : SSIM 가중치
    w_recon      : Reconstruction (Beer-Lambert) 가중치
    w_ortho      : Orthogonality 가중치
    w_smooth     : Smoothness (TV) 가중치
    """

    def __init__(self,
                 w_supervised : float = 1.0,
                 w_ssim       : float = 0.5,
                 w_recon      : float = 0.5,
                 w_ortho      : float = 0.3,
                 w_smooth     : float = 0.1):
        super().__init__()

        self.w_supervised = w_supervised
        self.w_ssim       = w_ssim
        self.w_recon      = w_recon
        self.w_ortho      = w_ortho
        self.w_smooth     = w_smooth

        # Beer-Lambert 흡수 계수 [R, G, B]
        mel_abs = torch.tensor([0.28, 0.18, 0.09]).view(1, 3, 1, 1)
        hem_abs = torch.tensor([0.10, 0.35, 0.05]).view(1, 3, 1, 1)
        self.register_buffer('mel_abs', mel_abs)
        self.register_buffer('hem_abs', hem_abs)

        self.ssim = SSIMLoss(window_size=11)

    @staticmethod
    def _masked_l1(pred, gt, mask):
        return (torch.abs(pred - gt) * mask).sum() / (mask.sum() + 1e-6)

    @staticmethod
    def _tv(x, mask):
        dh = torch.abs(x[:, :, 1:, :]  - x[:, :, :-1, :])
        dw = torch.abs(x[:, :, :, 1:]  - x[:, :, :, :-1])
        mh = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mw = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        return ((dh*mh).sum() / (mh.sum()+1e-6) +
                (dw*mw).sum() / (mw.sum()+1e-6))

    def forward(self,
                mel_pred  : torch.Tensor,   # [B, 1, H, W]
                hem_pred  : torch.Tensor,   # [B, 1, H, W]
                mel_gt    : torch.Tensor,   # [B, 1, H, W]
                hem_gt    : torch.Tensor,   # [B, 1, H, W]
                rgb_linear: torch.Tensor,   # [B, 3, H, W]
                face_mask : torch.Tensor,   # [B, 1, H, W]
                ) -> tuple:

        # 1. Supervised L1
        l_sup = (self._masked_l1(mel_pred, mel_gt, face_mask) +
                 self._masked_l1(hem_pred, hem_gt, face_mask))

        # 2. SSIM
        l_ssim = (self.ssim(mel_pred * face_mask, mel_gt * face_mask) +
                  self.ssim(hem_pred * face_mask, hem_gt * face_mask))

        # 3. Reconstruction (Beer-Lambert)
        od_recon  = mel_pred * self.mel_abs + hem_pred * self.hem_abs
        rgb_recon = torch.exp(-od_recon)
        mask_3ch  = face_mask.expand_as(rgb_recon)
        l_recon   = (torch.abs(rgb_recon - rgb_linear) * mask_3ch).sum() / \
                    (mask_3ch.sum() + 1e-6)

        # 4. Orthogonality
        l_ortho = (mel_pred * hem_pred * face_mask).sum() / \
                  (face_mask.sum() + 1e-6)

        # 5. Smoothness (TV)
        l_smooth = (self._tv(mel_pred, face_mask) +
                    self._tv(hem_pred, face_mask))

        total = (self.w_supervised * l_sup
               + self.w_ssim       * l_ssim
               + self.w_recon      * l_recon
               + self.w_ortho      * l_ortho
               + self.w_smooth     * l_smooth)

        detail = {
            'chroma_total': total.item(),
            'sup'         : l_sup.item(),
            'ssim'        : l_ssim.item(),
            'recon'       : l_recon.item(),
            'ortho'       : l_ortho.item(),
            'smooth'      : l_smooth.item(),
        }
        return total, detail


# ══════════════════════════════════════════════════════════════════════════════
# Ambient Invariant Feature Loss
# ══════════════════════════════════════════════════════════════════════════════
class AmbientInvariantLoss(nn.Module):
    """
    주변광 불변 Feature Consistency Loss

    원본과 조명변형 이미지의 DINOv2 feature가 동일해야 함
    → DINOv2가 주변광을 무시하는 표현을 학습
    → Decoder 연결 시 동일한 mel/hem 출력 보장

    방법
    ----
    'cosine'  : feature 방향 일치 (조명 세기 변화 허용)  ← 기본 추천
    'mse'     : feature 값 전체 일치 (강한 제약)
    'infonce' : Contrastive (같은 쌍은 가깝게, 다른 쌍은 멀게)

    Parameters
    ----------
    loss_type   : 'cosine' | 'mse' | 'infonce'
    temperature : InfoNCE 온도 파라미터 (낮을수록 강한 대조)
    use_patch   : True → patch token도 함께 사용 (더 강한 제약)
    patch_weight: patch token loss 가중치
    """

    def __init__(self,
                 loss_type   : str   = 'cosine',
                 temperature : float = 0.07,
                 use_patch   : bool  = False,
                 patch_weight: float = 0.5):
        super().__init__()
        self.loss_type    = loss_type
        self.temperature  = temperature
        self.use_patch    = use_patch
        self.patch_weight = patch_weight

    # ── Cosine ────────────────────────────────────────────────────────────────
    def _cosine(self,
                feat_orig: torch.Tensor,
                feat_aug : torch.Tensor) -> torch.Tensor:
        """
        1 - cosine_similarity → 0 으로 학습
        방향만 맞추므로 조명 세기 변화에 자연스럽게 강인

        feat: [B, D] or [B, N, D]
        """
        if feat_orig.dim() == 3:
            # patch tokens: [B, N, D] → [B*N, D]
            B, N, D  = feat_orig.shape
            feat_orig = feat_orig.reshape(B * N, D)
            feat_aug  = feat_aug.reshape(B * N, D)

        orig = F.normalize(feat_orig, dim=-1)
        aug  = F.normalize(feat_aug,  dim=-1)
        return 1 - (orig * aug).sum(dim=-1).mean()

    # ── MSE ───────────────────────────────────────────────────────────────────
    def _mse(self,
             feat_orig: torch.Tensor,
             feat_aug : torch.Tensor) -> torch.Tensor:
        """
        feature 값 자체를 일치 (방향 + 크기 모두)
        cosine보다 강한 제약 → 과도하면 학습 불안정
        """
        return F.mse_loss(feat_orig, feat_aug)

    # ── InfoNCE ───────────────────────────────────────────────────────────────
    def _infonce(self,
                 feat_orig: torch.Tensor,
                 feat_aug : torch.Tensor) -> torch.Tensor:
        """
        Contrastive Learning (NT-Xent)

        Positive pair : 같은 이미지의 (orig, aug) → 가깝게
        Negative pair : 다른 이미지  → 멀게

        feat_orig, feat_aug: [B, D] CLS token
        배치 크기가 클수록 효과적 (권장: B >= 32)
        """
        B = feat_orig.shape[0]

        # L2 normalize
        z_orig = F.normalize(feat_orig, dim=-1)   # [B, D]
        z_aug  = F.normalize(feat_aug,  dim=-1)   # [B, D]

        # [2B, D] → similarity matrix [2B, 2B]
        z_all  = torch.cat([z_orig, z_aug], dim=0)
        sim    = torch.mm(z_all, z_all.T) / self.temperature   # [2B, 2B]

        # 자기 자신 제외
        mask   = torch.eye(2 * B, device=sim.device).bool()
        sim    = sim.masked_fill(mask, float('-inf'))

        # positive pair 레이블
        # orig[i] ↔ aug[i] : i번째 ↔ B+i번째
        labels = torch.cat([
            torch.arange(B, 2*B, device=sim.device),
            torch.arange(0, B,   device=sim.device),
        ])

        return F.cross_entropy(sim, labels)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self,
                cls_orig  : torch.Tensor,              # [B, D]    CLS token
                cls_aug   : torch.Tensor,              # [B, D]    CLS token
                patch_orig: torch.Tensor = None,       # [B, N, D] patch tokens
                patch_aug : torch.Tensor = None,       # [B, N, D] patch tokens
                ) -> tuple:
        """
        Parameters
        ----------
        cls_orig   : 원본 이미지 CLS token feature    [B, D]
        cls_aug    : 조명변형 이미지 CLS token feature [B, D]
        patch_orig : 원본 patch token features        [B, N, D]  (선택)
        patch_aug  : 조명변형 patch token features    [B, N, D]  (선택)

        Returns
        -------
        total  : scalar loss
        detail : dict
        """
        # CLS token loss (전체 이미지 표현)
        if self.loss_type == 'cosine':
            loss_cls = self._cosine(cls_orig, cls_aug)
        elif self.loss_type == 'mse':
            loss_cls = self._mse(cls_orig, cls_aug)
        elif self.loss_type == 'infonce':
            loss_cls = self._infonce(cls_orig, cls_aug)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        total  = loss_cls
        detail = {
            'feat_total': loss_cls.item(),
            'feat_cls'  : loss_cls.item(),
            'feat_patch': 0.0,
        }

        # patch token loss (공간적 feature 일관성, 선택)
        if self.use_patch and patch_orig is not None and patch_aug is not None:
            if self.loss_type == 'cosine':
                loss_patch = self._cosine(patch_orig, patch_aug)
            elif self.loss_type == 'mse':
                loss_patch = self._mse(patch_orig, patch_aug)
            else:
                # InfoNCE는 patch token에 부적합 (N이 너무 큼)
                loss_patch = self._cosine(patch_orig, patch_aug)

            total               += self.patch_weight * loss_patch
            detail['feat_patch'] = loss_patch.item()
            detail['feat_total'] = total.item()

        return total, detail


# ══════════════════════════════════════════════════════════════════════════════
# Beer-Lambert Residual Illuminant Consistency Loss
# ══════════════════════════════════════════════════════════════════════════════
class IlluminantConsistencyLoss(nn.Module):
    """
    Beer-Lambert 잔차를 이용한 자기지도 조명 일관성 Loss

    핵심 원리
    ---------
    Beer-Lambert:  RGB = illuminant × exp(-OD)
    로그 도메인:   log(RGB) = log(illuminant) - OD
    잔차 정의:     R := log(RGB_measured) - log(RGB_recon)
                        = log(RGB_measured) + OD_recon
                        ≈ log(illuminant)

    → 잔차 R이 조명 추정치 역할. 실제 OD(chromophore)가 잘 분리될수록
      R의 패턴이 조명의 물리적 특성과 일치해야 함.

    Loss 구성
    ---------
    1. 공간 평활도 (TV): 조명은 공간적으로 서서히 변해야 함
    2. 스펙트럼 일관성: 중성(회색) 조명 가정 시 채널 간 잔차가 균등해야 함
       → 채널 간 편차 최소화
    3. 조명변형 뷰 일관성 (선택): 두 조명 뷰의 잔차 차이가
       공간적으로 균일한 패턴이어야 함 (illuminant ratio는 공간 무관)

    Parameters
    ----------
    w_smooth    : 공간 평활도 가중치
    w_spectral  : 스펙트럼 일관성 가중치
    w_augment   : 조명 뷰 간 일관성 가중치 (augmented view 없으면 무시)
    """

    # Beer-Lambert 흡수 계수 [R, G, B] — ChromophoreLoss와 동일해야 함
    _MEL_ABS = [0.28, 0.18, 0.09]
    _HEM_ABS = [0.10, 0.35, 0.05]

    def __init__(self,
                 w_smooth  : float = 1.0,
                 w_spectral: float = 0.5,
                 w_augment : float = 1.0):
        super().__init__()
        self.w_smooth   = w_smooth
        self.w_spectral = w_spectral
        self.w_augment  = w_augment

        mel_abs = torch.tensor(self._MEL_ABS).view(1, 3, 1, 1)
        hem_abs = torch.tensor(self._HEM_ABS).view(1, 3, 1, 1)
        self.register_buffer('mel_abs', mel_abs)
        self.register_buffer('hem_abs', hem_abs)

    def _residual(self,
                  log_rgb : torch.Tensor,    # [B, 3, H, W]
                  mel_pred: torch.Tensor,    # [B, 1, H, W]
                  hem_pred: torch.Tensor,    # [B, 1, H, W]
                  ) -> torch.Tensor:
        """
        Beer-Lambert 잔차 = log(illuminant) 추정치

        log(RGB_recon) = -OD_recon = -(mel_pred*mel_abs + hem_pred*hem_abs)
        잔차 = log_rgb − log_rgb_recon = log_rgb + OD_recon
        """
        od_recon  = mel_pred * self.mel_abs + hem_pred * self.hem_abs
        log_recon = -od_recon                         # log(RGB_recon)
        return log_rgb - log_recon                    # [B, 3, H, W]

    @staticmethod
    def _masked_tv(x: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """마스크 내 공간 평활도 (Total Variation)"""
        dh  = (x[:, :, 1:, :]  - x[:, :, :-1, :]).abs()
        dw  = (x[:, :, :, 1:]  - x[:, :, :, :-1]).abs()
        mh  = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mw  = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        tv  = (dh * mh).sum() / (mh.sum() + 1e-6)
        tv += (dw * mw).sum() / (mw.sum() + 1e-6)
        return tv

    def forward(self,
                log_rgb     : torch.Tensor,           # [B, 3, H, W]
                mel_pred    : torch.Tensor,           # [B, 1, H, W]
                hem_pred    : torch.Tensor,           # [B, 1, H, W]
                face_mask   : torch.Tensor,           # [B, 1, H, W]
                log_rgb_aug : torch.Tensor = None,    # [B, 3, H, W] (선택)
                mel_aug     : torch.Tensor = None,    # [B, 1, H, W] (선택)
                hem_aug     : torch.Tensor = None,    # [B, 1, H, W] (선택)
                ) -> tuple:
        """
        Parameters
        ----------
        log_rgb     : model.forward(return_feat=True)로 얻은 log(RGB)
        mel_pred    : 원본 뷰 melanin 예측
        hem_pred    : 원본 뷰 hemoglobin 예측
        face_mask   : [B, 1, H, W] 피부 영역 마스크
        log_rgb_aug : 조명 변형 뷰 log(RGB) (선택)
        mel_aug     : 조명 변형 뷰 melanin 예측 (선택)
        hem_aug     : 조명 변형 뷰 hemoglobin 예측 (선택)

        Returns
        -------
        total    : scalar loss
        detail   : dict
        residual : [B, 3, H, W]  log(illuminant) 추정치 (시각화/디버그용)
        """
        mask_3ch = face_mask.expand(-1, 3, -1, -1)   # [B, 3, H, W]

        # ── Beer-Lambert 잔차 ─────────────────────────────────────────────────
        residual = self._residual(log_rgb, mel_pred, hem_pred)   # [B, 3, H, W]

        # ── Loss 1: 공간 평활도 ───────────────────────────────────────────────
        l_smooth = self._masked_tv(residual, mask_3ch)

        # ── Loss 2: 스펙트럼 일관성 ───────────────────────────────────────────
        # 중성 조명: log(I_R) ≈ log(I_G) ≈ log(I_B)
        # 채널 간 편차 최소화 → 잔차가 채널에 무관하게 동일해야 함
        res_masked   = residual * mask_3ch                        # [B, 3, H, W]
        ch_mean      = res_masked.mean(dim=1, keepdim=True)       # [B, 1, H, W]
        ch_deviation = (res_masked - ch_mean).pow(2) * mask_3ch
        denom        = mask_3ch.sum() + 1e-6
        l_spectral   = ch_deviation.sum() / denom

        total  = self.w_smooth * l_smooth + self.w_spectral * l_spectral
        detail = {
            'illum_total'   : total.item(),
            'illum_smooth'  : l_smooth.item(),
            'illum_spectral': l_spectral.item(),
            'illum_augment' : 0.0,
        }

        # ── Loss 3: 조명변형 뷰 간 일관성 (선택) ─────────────────────────────
        if (log_rgb_aug is not None
                and mel_aug is not None
                and hem_aug is not None):
            residual_aug = self._residual(log_rgb_aug, mel_aug, hem_aug)

            # 두 잔차의 차이 = 두 조명 비율의 로그
            # 조명 비율은 공간에 무관 → 차이의 공간 분산이 0에 가까워야 함
            res_diff = (residual - residual_aug) * mask_3ch        # [B, 3, H, W]
            # 픽셀 평균 제거 후 분산
            diff_mean = res_diff.sum(dim=[2, 3], keepdim=True) / \
                        (mask_3ch.sum(dim=[2, 3], keepdim=True) + 1e-6)
            l_aug = ((res_diff - diff_mean).pow(2) * mask_3ch).sum() / denom

            total                  += self.w_augment * l_aug
            detail['illum_augment'] = l_aug.item()
            detail['illum_total']   = total.item()

        return total, detail, residual


# ══════════════════════════════════════════════════════════════════════════════
# 학습 단계별 Loss 가중치 스케줄러
# ══════════════════════════════════════════════════════════════════════════════
def get_loss_weights(epoch: int, total_epochs: int) -> dict:
    """
    학습 단계에 따라 Loss 가중치 조절

    Phase 1 (0~30%)  : Supervised + Recon → 빠른 수렴
    Phase 2 (30~70%) : SSIM + Ortho + feat consistency 추가
    Phase 3 (70~100%): 전체 Loss 적용

    Usage
    -----
    weights = get_loss_weights(epoch, total_epochs)
    for k, v in weights.items():
        setattr(criterion, k, v)
    """
    progress = epoch / total_epochs

    if progress < 0.3:
        return dict(
            w_supervised = 1.0, w_ssim  = 0.0,
            w_recon      = 0.5, w_ortho = 0.0, w_smooth = 0.0
        )
    elif progress < 0.7:
        return dict(
            w_supervised = 1.0, w_ssim  = 0.3,
            w_recon      = 0.5, w_ortho = 0.2, w_smooth = 0.1
        )
    else:
        return dict(
            w_supervised = 1.0, w_ssim  = 0.5,
            w_recon      = 0.5, w_ortho = 0.3, w_smooth = 0.1
        )
