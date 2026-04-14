"""
model.py
========
ChromophoreNet
  LogAbsorbanceInput (trainable): RGB → log-absorbance + channel ratio → DINOv2 입력
  DINOv2 (frozen)   : 주변광 불변 feature 추출
  Bridge (trainable): DINOv2 → Decoder 형식 변환  [InstanceNorm2d]
  Decoder (trainable): feature → melanin / hemoglobin

입력 : RGB linear [B, 3, H, W]
출력 : mel_pred   [B, 1, H, W]
       hem_pred   [B, 1, H, W]
       (선택) cls_feat   [B, 768]      ← AmbientInvariantLoss용
       (선택) patch_feat [B, N, 768]   ← AmbientInvariantLoss용
       (선택) log_rgb    [B, 3, H, W]  ← IlluminantConsistencyLoss용

변경 사항 (illumination robustness)
------------------------------------
1. LogAbsorbanceInput : RGB → -log(RGB) [OD] + log-ratio 채널
   - Beer-Lambert: log(RGB) = log(illuminant) - OD  → 조명이 additive offset
   - 채널 log-ratio (log(R/G), log(G/B)): 조명 독립적
   - 5ch → 3ch 학습 가능 projection으로 DINOv2 입력 생성
2. InstanceNorm2d (affine=True): Bridge refine conv 후 배치 통계 대신
   인스턴스 통계로 정규화 → 이미지별 전역 조명 offset 제거
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ══════════════════════════════════════════════════════════════════════════════
# Log-Absorbance Input Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
class LogAbsorbanceInput(nn.Module):
    """
    RGB linear → log-absorbance 도메인 전처리

    Beer-Lambert 법칙:
        RGB = illuminant × exp(-OD)
        log(RGB) = log(illuminant) − OD

    조명이 log 도메인에서 additive offset이 되므로:
      - InstanceNorm이 global offset(조명)을 효과적으로 제거
      - 채널 차이(log-ratio)는 중성 조명 가정 시 조명에 무관

    입력 [B, 3, H, W] (linear RGB, 0~1)
    출력
      dino_in  [B, 3, H, W] : DINOv2 입력 (학습 가능 5→3 projection)
      log_rgb  [B, 3, H, W] : log(RGB+ε), IlluminantConsistencyLoss용

    Parameters
    ----------
    eps : log 연산 수치 안정성 (기본 1e-6)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # OD 3ch + log-ratio 2ch → DINOv2 입력 3ch
        self.proj = nn.Conv2d(5, 3, kernel_size=1, bias=True)
        self._init_proj()

    def _init_proj(self):
        """초기값: OD 채널을 그대로 통과하도록 설정"""
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            # OD 채널 (0,1,2) → 출력 채널 (0,1,2) 직접 연결
            self.proj.weight[0, 0, 0, 0] = 1.0
            self.proj.weight[1, 1, 0, 0] = 1.0
            self.proj.weight[2, 2, 0, 0] = 1.0
            nn.init.zeros_(self.proj.bias)

    def forward(self, rgb: torch.Tensor) -> tuple:
        """
        Returns
        -------
        dino_in : [B, 3, H, W]  DINOv2 입력
        log_rgb : [B, 3, H, W]  log(RGB)  ← IlluminantConsistencyLoss에 전달
        """
        rgb_safe = rgb.clamp(min=self.eps)
        log_rgb  = torch.log(rgb_safe)             # [B, 3, H, W]
        od       = -log_rgb                         # optical density estimate

        # 채널 log-ratio: 중성 조명 가정 시 조명 독립
        rg = log_rgb[:, 0:1, :, :] - log_rgb[:, 1:2, :, :]   # log(R/G)
        gb = log_rgb[:, 1:2, :, :] - log_rgb[:, 2:3, :, :]   # log(G/B)

        feat5   = torch.cat([od, rg, gb], dim=1)   # [B, 5, H, W]
        dino_in = self.proj(feat5)                  # [B, 3, H, W]

        return dino_in, log_rgb


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
        # InstanceNorm2d: 이미지별 통계 정규화 → 전역 조명 offset 제거
        # affine=True: 학습 가능한 scale/shift 유지
        self.refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.InstanceNorm2d(ch, affine=True),
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

        # ── Log-Absorbance Preprocessing ──────────────────────────────────────
        # RGB → OD + log-ratio → DINOv2 입력 (학습 가능)
        self.log_input = LogAbsorbanceInput(eps=1e-6)

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
        rgb         : [B, 3, H, W] linear RGB (0~1)
        return_feat : True → feature + log_rgb 반환
                      (AmbientInvariantLoss / IlluminantConsistencyLoss용)

        Returns
        -------
        return_feat=False:
          mel_pred, hem_pred

        return_feat=True:
          mel_pred, hem_pred, cls_feat, patch_feat, log_rgb

          cls_feat   : [B, 768]    CLS token
          patch_feat : [B, N, 768] patch tokens
          log_rgb    : [B, 3, H, W] log(RGB) ← IlluminantConsistencyLoss에 전달
        """
        # Log-Absorbance 전처리
        # dino_in: OD + log-ratio 기반 DINOv2 입력
        # log_rgb: Beer-Lambert 잔차 계산용
        dino_in, log_rgb = self.log_input(rgb)

        # DINOv2: feature 추출 (log-absorbance 입력)
        dino_out      = self.dinov2(dino_in, output_hidden_states=True)
        hidden_states = dino_out.hidden_states   # tuple [B, N+1, 768]

        # Bridge: 형식 변환 (InstanceNorm2d)
        features = self.bridge(hidden_states)

        # Decoder: mel/hem 분리
        mel_pred, hem_pred = self.decoder(features)

        if return_feat:
            last_hidden = hidden_states[-1]          # [B, N+1, 768]
            cls_feat    = last_hidden[:, 0,  :]      # [B, 768]
            patch_feat  = last_hidden[:, 1:, :]      # [B, N, 768]
            return mel_pred, hem_pred, cls_feat, patch_feat, log_rgb

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

    log_params = sum(p.numel() for p in model.log_input.parameters())
    print(f"\n=== 모델 파라미터 ===")
    print(f"  DINOv2  (frozen)        : {(total-trainable)/1e6:.1f}M")
    print(f"  LogAbsorbanceInput      : {log_params/1e3:.1f}K  (5→3 projection)")
    print(f"  Bridge (InstanceNorm)   : ")
    print(f"  Bridge + Decoder + Log  : {trainable/1e6:.1f}M  ← 학습 대상")
    print(f"  전체                    : {total/1e6:.1f}M\n")

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
