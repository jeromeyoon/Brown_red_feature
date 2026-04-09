"""
dino-UNet/model.py
==================
DinoUNet
  DINOv2 (frozen / fine-tune) : 멀티스케일 feature 추출
  UNetDecoder (trainable)     : skip connection 기반 mel/hem 예측

기존 ChromophoreNet과의 차이
---------------------------
ChromophoreNet  : DINOv2 → Bridge → 외부 사전학습 Decoder (별도 .pth 필요)
DinoUNet        : DINOv2 → 내장 UNetDecoder (처음부터 end-to-end 학습 가능)
                  외부 Decoder checkpoint 불필요

아키텍처
--------
입력 RGB [B, 3, H, W]
  ↓
DINOv2 (ViT-B)
  ↓ layer 3, 6, 9, 12 에서 멀티스케일 피처 추출
  ↓
FeatureAdapter  : ViT 토큰 → 공간 피처맵 변환 (형식 변환)
  ↓
UNetDecoder     : 4단계 UpBlock + skip connection
  ↓
출력 mel_pred [B, 1, H, W], hem_pred [B, 1, H, W]

입력 : RGB linear [B, 3, H, W]
출력 : mel_pred   [B, 1, H, W]
       hem_pred   [B, 1, H, W]
       (선택) cls_feat   [B, 768]    ← AmbientInvariantLoss용
       (선택) patch_feat [B, N, 768] ← AmbientInvariantLoss용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ══════════════════════════════════════════════════════════════════════════════
# 기본 빌딩 블록
# ══════════════════════════════════════════════════════════════════════════════
class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU"""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    U-Net 디코더 업샘플링 블록

    1. ConvTranspose2d × 2 (2배 업샘플)
    2. Skip connection concat
    3. ConvBnRelu × 2

    in_ch  : bottleneck / 이전 업블록 출력 채널
    skip_ch: skip connection 채널 (FeatureAdapter 출력)
    out_ch : 이 블록 출력 채널
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                           kernel_size=2, stride=2)
        self.conv1 = ConvBnRelu(in_ch // 2 + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # 크기 불일치 보정 (홀수 해상도 대응)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Feature Adapter : ViT 토큰 → 공간 피처맵
# ══════════════════════════════════════════════════════════════════════════════
class FeatureAdapter(nn.Module):
    """
    DINOv2 중간 레이어 토큰 → 멀티스케일 공간 피처맵 변환

    ChromophoreNet의 DINOv2Bridge와 동일한 역할
    (형식 변환 전담, feature 추출은 DINOv2가 담당)

    layer 3  → bottleneck (1/32)
    layer 6  → skip3      (1/16)
    layer 9  → skip2      (1/8)
    layer 12 → skip1      (1/4)
    """

    LAYER_INDICES = [3, 6, 9, 12]

    def __init__(self,
                 dino_dim : int  = 768,
                 dec_chs  : list = [512, 256, 128, 64],
                 img_size : int  = 224):
        super().__init__()

        self.target_sizes = [
            (img_size // 32, img_size // 32),
            (img_size // 16, img_size // 16),
            (img_size // 8,  img_size // 8),
            (img_size // 4,  img_size // 4),
        ]

        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dino_dim),
                nn.Linear(dino_dim, ch),
            )
            for ch in dec_chs
        ])

        self.refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )
            for ch in dec_chs
        ])

    def _to_spatial(self, tokens: torch.Tensor,
                    size: tuple) -> torch.Tensor:
        """[B, N, C] → [B, C, H, W]"""
        B, N, C = tokens.shape
        h = w   = int(N ** 0.5)
        x = tokens.permute(0, 2, 1).reshape(B, C, h, w)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, hidden_states: tuple) -> list:
        """
        Returns
        -------
        [bottleneck, skip3, skip2, skip1]  (채널: dec_chs 순서)
        """
        features = []
        for idx, proj, refine, size in zip(
            self.LAYER_INDICES, self.projs, self.refines, self.target_sizes
        ):
            tokens = hidden_states[idx][:, 1:, :]   # CLS 제거
            tokens = proj(tokens)
            x      = self._to_spatial(tokens, size)
            x      = refine(x)
            features.append(x)
        return features


# ══════════════════════════════════════════════════════════════════════════════
# U-Net Decoder
# ══════════════════════════════════════════════════════════════════════════════
class UNetDecoder(nn.Module):
    """
    4단계 U-Net 디코더

    ChromophoreNet의 외부 Decoder와 달리
    → 사전학습 checkpoint 불필요
    → skip connection이 명시적으로 구현됨

    dec_chs = [512, 256, 128, 64] 기준:
      bottleneck(512) → UpBlock → 256
      skip3(256)      → UpBlock → 128
      skip2(128)      → UpBlock → 64
      skip1(64)       → UpBlock → 32
      최종 Conv → mel(1ch) + hem(1ch)
    """

    def __init__(self, dec_chs: list = [512, 256, 128, 64]):
        super().__init__()

        b, s3, s2, s1 = dec_chs   # [512, 256, 128, 64]

        self.up1 = UpBlock(b,  s3, s3 // 2)    # 512 → 128
        self.up2 = UpBlock(s3 // 2 * 2, s2, s2 // 2)  # needs adjustment
        self.up3 = UpBlock(s2 // 2 * 2, s1, s1 // 2)

        # up1: in=512 → upsample to 256, cat skip3(256) → conv to 256
        # up2: in=256 → upsample to 128, cat skip2(128) → conv to 128
        # up3: in=128 → upsample to 64,  cat skip1(64)  → conv to 64

        self.up1 = UpBlock(b,       s3, s3)   # bottleneck(512) + skip3(256)
        self.up2 = UpBlock(s3,      s2, s2)   # up1_out(256)    + skip2(128)
        self.up3 = UpBlock(s2,      s1, s1)   # up2_out(128)    + skip1(64)
        self.up4 = nn.Sequential(             # 최종 2배 업샘플 (no skip)
            nn.ConvTranspose2d(s1, s1 // 2, kernel_size=2, stride=2),
            ConvBnRelu(s1 // 2, s1 // 2),
        )

        self.head_mel = nn.Conv2d(s1 // 2, 1, kernel_size=1)
        self.head_hem = nn.Conv2d(s1 // 2, 1, kernel_size=1)

    def forward(self, features: list) -> tuple:
        """
        features : [bottleneck, skip3, skip2, skip1]
                   FeatureAdapter 출력 순서와 동일

        Returns
        -------
        mel_pred [B, 1, H, W], hem_pred [B, 1, H, W]
        """
        bottleneck, skip3, skip2, skip1 = features

        x = self.up1(bottleneck, skip3)   # 1/32 → 1/16
        x = self.up2(x,          skip2)   # 1/16 → 1/8
        x = self.up3(x,          skip1)   # 1/8  → 1/4
        x = self.up4(x)                   # 1/4  → 1/2

        # 최종 입력 해상도로 업샘플 (H, W 복원)
        # bottleneck 해상도가 img_size//32이므로 4단계 업샘플로 img_size//2까지
        # 나머지 2배는 interpolate로 처리
        mel_pred = self.head_mel(x)
        hem_pred = self.head_hem(x)

        return mel_pred, hem_pred


# ══════════════════════════════════════════════════════════════════════════════
# Main Model
# ══════════════════════════════════════════════════════════════════════════════
class DinoUNet(nn.Module):
    """
    DINOv2 + U-Net Decoder for Chromophore Prediction

    ChromophoreNet과의 차이
    ----------------------
    ChromophoreNet:
      - 외부 사전학습 Decoder 필요 (별도 .pth 로드)
      - Bridge가 형식만 변환, Decoder는 분리된 모듈
    DinoUNet:
      - Decoder 내장 (checkpoint 불필요)
      - FeatureAdapter + UNetDecoder 통합 구조
      - skip connection이 명시적으로 구현됨
      - Phase 1/2 학습 전략 내장

    Parameters
    ----------
    dinov2_path  : 로컬 DINOv2 경로 (AutoModel.from_pretrained)
    dec_chs      : 멀티스케일 채널 [bottleneck, skip3, skip2, skip1]
    img_size     : 입력 패치 크기
    freeze_dino  : True → DINOv2 전체 frozen (Phase 1)
                   False → 전체 end-to-end (Phase 2)
    unfreeze_last: freeze_dino=False 시 마지막 N 블록만 해제
                   None → 전체 해제
    """

    def __init__(self,
                 dinov2_path  : str,
                 dec_chs      : list = [512, 256, 128, 64],
                 img_size     : int  = 224,
                 freeze_dino  : bool = True,
                 unfreeze_last: int  = None):
        super().__init__()

        # ── DINOv2 ────────────────────────────────────────────────────────────
        self.dinov2 = AutoModel.from_pretrained(
            dinov2_path, local_files_only=True
        )
        self._setup_freeze(freeze_dino, unfreeze_last)

        # ── Feature Adapter ───────────────────────────────────────────────────
        self.adapter = FeatureAdapter(
            dino_dim = 768,
            dec_chs  = dec_chs,
            img_size = img_size,
        )

        # ── U-Net Decoder ─────────────────────────────────────────────────────
        self.decoder = UNetDecoder(dec_chs=dec_chs)

    def _setup_freeze(self, freeze_dino: bool, unfreeze_last: int):
        """DINOv2 freeze 설정"""
        # 전체 freeze
        for p in self.dinov2.parameters():
            p.requires_grad = False

        if not freeze_dino:
            if unfreeze_last is None:
                # 전체 해제
                for p in self.dinov2.parameters():
                    p.requires_grad = True
            else:
                # 마지막 N 블록만 해제 (Phase 2 fine-tuning 권장)
                encoder_layers = self.dinov2.encoder.layer
                for layer in encoder_layers[-unfreeze_last:]:
                    for p in layer.parameters():
                        p.requires_grad = True

    def unfreeze_dino(self, last_n_blocks: int = None):
        """
        Phase 2 진입 시 호출: DINOv2 일부/전체 해제

        Usage
        -----
        model.unfreeze_dino(last_n_blocks=4)  # 마지막 4블록 해제
        model.unfreeze_dino()                 # 전체 해제
        """
        self._setup_freeze(freeze_dino=False, unfreeze_last=last_n_blocks)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  DINOv2 unfreeze 완료 → 학습 파라미터: {trainable/1e6:.1f}M")

    def forward(self,
                rgb         : torch.Tensor,
                return_feat : bool = False):
        """
        Parameters
        ----------
        rgb         : [B, 3, H, W] linear RGB
        return_feat : True → CLS/patch feature 반환 (AmbientInvariantLoss용)

        Returns
        -------
        return_feat=False:
          mel_pred, hem_pred

        return_feat=True:
          mel_pred, hem_pred, cls_feat [B,768], patch_feat [B,N,768]
        """
        dino_out      = self.dinov2(rgb, output_hidden_states=True)
        hidden_states = dino_out.hidden_states   # tuple [B, N+1, 768]

        features = self.adapter(hidden_states)   # [bottleneck, s3, s2, s1]

        mel_pred, hem_pred = self.decoder(features)

        # 입력 해상도 복원
        mel_pred = F.interpolate(mel_pred, size=rgb.shape[2:],
                                 mode='bilinear', align_corners=False)
        hem_pred = F.interpolate(hem_pred, size=rgb.shape[2:],
                                 mode='bilinear', align_corners=False)

        if return_feat:
            last_hidden = hidden_states[-1]
            cls_feat    = last_hidden[:, 0,  :]   # [B, 768]
            patch_feat  = last_hidden[:, 1:, :]   # [B, N, 768]
            return mel_pred, hem_pred, cls_feat, patch_feat

        return mel_pred, hem_pred


# ══════════════════════════════════════════════════════════════════════════════
# Builder
# ══════════════════════════════════════════════════════════════════════════════
def build_dino_unet(dinov2_path  : str,
                    dec_chs      : list = [512, 256, 128, 64],
                    img_size     : int  = 224,
                    freeze_dino  : bool = True,
                    unfreeze_last: int  = None) -> DinoUNet:
    """
    DinoUNet 생성 + 파라미터 통계 출력

    ChromophoreNet의 build_model과 동일한 인터페이스
    단, decoder_ckpt / decoder_cls 인자 불필요
    """
    model = DinoUNet(
        dinov2_path   = dinov2_path,
        dec_chs       = dec_chs,
        img_size      = img_size,
        freeze_dino   = freeze_dino,
        unfreeze_last = unfreeze_last,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    frozen    = total - trainable

    print(f"\n=== DinoUNet 파라미터 ===")
    print(f"  DINOv2  ({'frozen' if freeze_dino else 'trainable'}) : {frozen/1e6:.1f}M")
    print(f"  Adapter + UNetDecoder : {trainable/1e6:.1f}M  ← 학습 대상")
    print(f"  전체                  : {total/1e6:.1f}M\n")

    return model
