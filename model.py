"""
model.py
========
ChromophoreNet
  DINOv2 (frozen)   : 주변광 불변 feature 추출
  Bridge (trainable): DINOv2 → Decoder 형식 변환
  Decoder (trainable): feature → melanin / hemoglobin

입력 : RGB linear [B, 3, H, W]
출력 : mel_pred   [B, 1, H, W]
       hem_pred   [B, 1, H, W]
       (선택) cls_feat   [B, 768]      ← AmbientInvariantLoss용
       (선택) patch_feat [B, N, 768]   ← AmbientInvariantLoss용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ══════════════════════════════════════════════════════════════════════════════
# Bridge
# ══════════════════════════════════════════════════════════════════════════════
class DINOv2Bridge(nn.Module):
    """
    DINOv2 patch tokens → Decoder multi-scale spatial feature 변환

    역할: feature 추출 X / 형식 변환 O
      [B, N, 768] → [B, dec_ch, H, W] × 4 스케일

    ViT-B 기준: layer 3, 6, 9, 12 사용
    """

    def __init__(self,
                 dino_dim : int  = 768,
                 dec_chs  : list = [512, 256, 128, 64],
                 img_size : int  = 224):
        super().__init__()

        self.target_sizes = [
            (img_size // 32, img_size // 32),   # bottleneck
            (img_size // 16, img_size // 16),   # skip3
            (img_size // 8,  img_size // 8),    # skip2
            (img_size // 4,  img_size // 4),    # skip1
        ]

        # sequence → channel projection (LN → Linear)
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dino_dim),
                nn.Linear(dino_dim, ch),
            )
            for ch in dec_chs
        ])

        # spatial refinement conv
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
        hidden_states : DINOv2 모든 레이어 출력 tuple
        returns       : [bottleneck, skip3, skip2, skip1]
        """
        layer_indices = [3, 6, 9, 12]
        features = []

        for idx, proj, refine, size in zip(
            layer_indices, self.projs, self.refines, self.target_sizes
        ):
            tokens = hidden_states[idx][:, 1:, :]   # CLS 제거 [B, N, 768]
            tokens = proj(tokens)                    # [B, N, dec_ch]
            x      = self._to_spatial(tokens, size)  # [B, dec_ch, H, W]
            x      = refine(x)
            features.append(x)

        return features


# ══════════════════════════════════════════════════════════════════════════════
# Main Model
# ══════════════════════════════════════════════════════════════════════════════
class ChromophoreNet(nn.Module):
    """
    학습 목표
    ---------
    DINOv2 : 주변광 변화에 강인한 feature 생성
             (AmbientInvariantLoss로 feature consistency 학습)
    Bridge : DINOv2 feature → Decoder 입력 형식 변환
    Decoder: feature → melanin / hemoglobin 분리

    Parameters
    ----------
    dinov2_path  : 로컬 DINOv2 경로
    decoder      : pretrained Decoder 모듈
    dec_chs      : Bridge → Decoder 채널 [bottleneck, skip3, skip2, skip1]
    img_size     : 입력 이미지 크기 (patch_size와 동일)
    freeze_dino  : DINOv2 frozen 여부
    """

    def __init__(self,
                 dinov2_path : str,
                 decoder     : nn.Module,
                 dec_chs     : list = [512, 256, 128, 64],
                 img_size    : int  = 224,
                 freeze_dino : bool = True):
        super().__init__()

        # ── DINOv2 ────────────────────────────────────────────────────────────
        self.dinov2 = AutoModel.from_pretrained(
            dinov2_path, local_files_only=True
        )
        if freeze_dino:
            for p in self.dinov2.parameters():
                p.requires_grad = False

        # ── Bridge ────────────────────────────────────────────────────────────
        self.bridge = DINOv2Bridge(
            dino_dim = 768,
            dec_chs  = dec_chs,
            img_size = img_size,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = decoder

    def forward(self,
                rgb         : torch.Tensor,
                return_feat : bool = False):
        """
        Parameters
        ----------
        rgb         : [B, 3, H, W] linear RGB
        return_feat : True → feature도 반환 (AmbientInvariantLoss용)

        Returns
        -------
        return_feat=False:
          mel_pred, hem_pred

        return_feat=True:
          mel_pred, hem_pred, cls_feat, patch_feat

          cls_feat   : [B, 768]    CLS token  (전체 이미지 표현)
          patch_feat : [B, N, 768] patch tokens (공간 표현)
        """
        # DINOv2: feature 추출
        dino_out      = self.dinov2(rgb, output_hidden_states=True)
        hidden_states = dino_out.hidden_states  # tuple [B, N+1, 768]

        # Bridge: 형식 변환
        features = self.bridge(hidden_states)

        # Decoder: mel/hem 분리
        mel_pred, hem_pred = self.decoder(features)

        if return_feat:
            # 마지막 레이어 feature 반환
            last_hidden  = hidden_states[-1]           # [B, N+1, 768]
            cls_feat     = last_hidden[:, 0,  :]       # [B, 768]
            patch_feat   = last_hidden[:, 1:, :]       # [B, N, 768]
            return mel_pred, hem_pred, cls_feat, patch_feat

        return mel_pred, hem_pred


# ══════════════════════════════════════════════════════════════════════════════
# Builder
# ══════════════════════════════════════════════════════════════════════════════
def build_model(dinov2_path : str,
                decoder     : nn.Module,
                dec_chs     : list = [512, 256, 128, 64],
                img_size    : int  = 224,
                freeze_dino : bool = True) -> ChromophoreNet:
    """
    ChromophoreNet 생성 + 파라미터 통계 출력

    dec_chs 확인:
      inspect_decoder_channels('./pretrained/decoder.pth', 'decoder')
    """
    model = ChromophoreNet(
        dinov2_path = dinov2_path,
        decoder     = decoder,
        dec_chs     = dec_chs,
        img_size    = img_size,
        freeze_dino = freeze_dino,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    print(f"\n=== 모델 파라미터 ===")
    print(f"  DINOv2  (frozen)  : {(total-trainable)/1e6:.1f}M")
    print(f"  Bridge + Decoder  : {trainable/1e6:.1f}M  ← 학습 대상")
    print(f"  전체              : {total/1e6:.1f}M\n")

    return model


def inspect_decoder_channels(ckpt_path     : str,
                              decoder_prefix: str = 'decoder'):
    """Decoder 채널 구조 확인 (dec_chs 설정용)"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd   = ckpt.get('state_dict', ckpt.get('model', ckpt))
    print(f"=== Decoder 채널 구조 ({decoder_prefix}.*) ===")
    for k, v in sd.items():
        if k.startswith(decoder_prefix) and 'weight' in k and v.dim() == 4:
            print(f"  {k}: in={v.shape[1]}, out={v.shape[0]}")
