"""
train.py
========
ChromophoreNet 학습 메인 스크립트

전체 Loss:
  total = loss_chroma + w_feat * loss_feat

  loss_chroma : melanin/hemoglobin 분리 Loss
  loss_feat   : 주변광 불변 feature consistency Loss
                원본 feature vs 조명변형 feature → 동일해야 함

실행:
  python train.py
  nohup python train.py > train.log 2>&1 &
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from model      import build_model, inspect_decoder_channels
from loss       import ChromophoreLoss, AmbientInvariantLoss, get_loss_weights
from dataset    import PatchDataset
from ambient_aug import get_ambient_transform, apply_ambient_aug


# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # 경로
    patch_dir       = './patches',                 # extract_patches.py 출력
    dinov2_path     = './pretrained/dinov2-base',  # ← 수정
    decoder_ckpt    = './pretrained/decoder.pth',  # ← 수정
    decoder_prefix  = 'decoder',
    checkpoint_dir  = './checkpoints',

    # 모델
    dec_chs         = [512, 256, 128, 64],   # ← Decoder 채널 확인 후 수정
    freeze_dino     = True,
    img_size        = 224,                   # patch_size와 동일

    # 학습
    epochs          = 50,
    batch_size      = 16,
    lr              = 1e-4,
    weight_decay    = 1e-4,
    num_workers     = 8,

    # Feature Consistency Loss 설정
    feat_loss_type  = 'cosine',   # 'cosine' | 'mse' | 'infonce'
    feat_temperature= 0.07,       # InfoNCE 온도 (cosine/mse는 무시)
    use_patch_feat  = False,      # patch token도 consistency loss에 사용
    patch_feat_w    = 0.5,        # patch token loss 가중치
    w_feat          = 0.5,        # feature consistency loss 전체 가중치

    # Chromophore Loss 초기 가중치 (get_loss_weights로 단계적 조절)
    w_supervised    = 1.0,
    w_ssim          = 0.0,
    w_recon         = 0.5,
    w_ortho         = 0.0,
    w_smooth        = 0.0,
)


# ══════════════════════════════════════════════════════════════════════════════
# Decoder 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_pretrained_decoder(ckpt_path     : str,
                             decoder_cls  ,
                             decoder_prefix: str = 'decoder') -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd   = ckpt.get('state_dict', ckpt.get('model', ckpt))

    decoder_sd = {
        k[len(decoder_prefix) + 1:]: v
        for k, v in sd.items()
        if k.startswith(decoder_prefix + '.')
    }

    decoder = decoder_cls()
    missing, unexpected = decoder.load_state_dict(decoder_sd, strict=False)
    if missing:
        print(f"⚠️  Missing:    {missing}")
    if unexpected:
        print(f"⚠️  Unexpected: {unexpected}")
    print("✅ Decoder weight 로드 완료")
    return decoder


# ══════════════════════════════════════════════════════════════════════════════
# 조명변형 이미지 생성 (배치 단위)
# ══════════════════════════════════════════════════════════════════════════════
def make_ambient_batch(rgb        : torch.Tensor,
                       transform  ,
                       device     : torch.device) -> torch.Tensor:
    """
    원본 RGB 배치로부터 조명변형 배치 생성

    Parameters
    ----------
    rgb       : [B, 3, H, W] float32 tensor (linear RGB)
    transform : get_ambient_transform() 반환값
    device    : 결과 tensor device

    Returns
    -------
    rgb_aug : [B, 3, H, W] float32 tensor (조명변형)
    """
    # tensor → numpy [B, H, W, 3]
    rgb_np  = rgb.cpu().permute(0, 2, 3, 1).numpy()

    # 조명 augmentation 적용
    aug_np  = apply_ambient_aug(transform, rgb_np)  # [B, H, W, 3]

    # numpy → tensor [B, 3, H, W]
    rgb_aug = torch.from_numpy(
        aug_np.transpose(0, 3, 1, 2)
    ).float().to(device)

    return rgb_aug


# ══════════════════════════════════════════════════════════════════════════════
# 학습 / 검증
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, chroma_loss, feat_loss,
                    ambient_transform, optimizer, device, cfg):
    model.train()

    total_loss   = 0.0
    chroma_log   = {k: 0.0 for k in ['sup', 'ssim', 'recon', 'ortho', 'smooth']}
    feat_log     = {'feat_cls': 0.0, 'feat_patch': 0.0}

    for batch in loader:
        rgb       = batch['rgb'].to(device)         # [B, 3, H, W]
        mel_gt    = batch['mel_gt'].to(device)
        hem_gt    = batch['hem_gt'].to(device)
        face_mask = batch['face_mask'].to(device)

        # ── 조명변형 이미지 생성 ─────────────────────────────────────────────
        rgb_aug = make_ambient_batch(rgb, ambient_transform, device)

        # ── Forward: 원본 (mel/hem + feature 반환) ───────────────────────────
        mel_pred, hem_pred, cls_orig, patch_orig = model(
            rgb, return_feat=True
        )

        # ── Forward: 조명변형 (feature만 사용) ──────────────────────────────
        with torch.no_grad() if cfg['freeze_dino'] else torch.enable_grad():
            _, _, cls_aug, patch_aug = model(rgb_aug, return_feat=True)

        # ── Chromophore Loss (mel/hem 분리) ──────────────────────────────────
        loss_c, detail_c = chroma_loss(
            mel_pred   = mel_pred,
            hem_pred   = hem_pred,
            mel_gt     = mel_gt,
            hem_gt     = hem_gt,
            rgb_linear = rgb,
            face_mask  = face_mask,
        )

        # ── Feature Consistency Loss (주변광 불변) ───────────────────────────
        loss_f, detail_f = feat_loss(
            cls_orig   = cls_orig,
            cls_aug    = cls_aug,
            patch_orig = patch_orig if cfg['use_patch_feat'] else None,
            patch_aug  = patch_aug  if cfg['use_patch_feat'] else None,
        )

        # ── 합산 ────────────────────────────────────────────────────────────
        total = loss_c + cfg['w_feat'] * loss_f

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total.item()
        for k in chroma_log:
            chroma_log[k] += detail_c.get(k, 0.0)
        for k in feat_log:
            feat_log[k] += detail_f.get(k, 0.0)

    n = len(loader)
    log = {
        'total'      : total_loss / n,
        **{k: v / n for k, v in chroma_log.items()},
        **{k: v / n for k, v in feat_log.items()},
    }
    return log


@torch.no_grad()
def validate(model, loader, chroma_loss, feat_loss,
             ambient_transform, device, cfg):
    model.eval()

    total_loss = 0.0
    chroma_log = {k: 0.0 for k in ['sup', 'ssim', 'recon', 'ortho', 'smooth']}
    feat_log   = {'feat_cls': 0.0, 'feat_patch': 0.0}

    for batch in loader:
        rgb       = batch['rgb'].to(device)
        mel_gt    = batch['mel_gt'].to(device)
        hem_gt    = batch['hem_gt'].to(device)
        face_mask = batch['face_mask'].to(device)

        rgb_aug = make_ambient_batch(rgb, ambient_transform, device)

        mel_pred, hem_pred, cls_orig, patch_orig = model(
            rgb, return_feat=True
        )
        _, _, cls_aug, patch_aug = model(rgb_aug, return_feat=True)

        loss_c, detail_c = chroma_loss(
            mel_pred, hem_pred, mel_gt, hem_gt, rgb, face_mask
        )
        loss_f, detail_f = feat_loss(
            cls_orig   = cls_orig,
            cls_aug    = cls_aug,
            patch_orig = patch_orig if cfg['use_patch_feat'] else None,
            patch_aug  = patch_aug  if cfg['use_patch_feat'] else None,
        )

        total = loss_c + cfg['w_feat'] * loss_f
        total_loss += total.item()

        for k in chroma_log:
            chroma_log[k] += detail_c.get(k, 0.0)
        for k in feat_log:
            feat_log[k] += detail_f.get(k, 0.0)

    n = len(loader)
    log = {
        'total'      : total_loss / n,
        **{k: v / n for k, v in chroma_log.items()},
        **{k: v / n for k, v in feat_log.items()},
    }
    return log


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    Path(CFG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # ── Decoder 채널 확인 ──────────────────────────────────────────────────────
    inspect_decoder_channels(CFG['decoder_ckpt'], CFG['decoder_prefix'])

    # ── Decoder 로드 ───────────────────────────────────────────────────────────
    # TODO: 실제 Decoder 클래스로 교체 후 아래 raise 제거
    # from your_decoder import YourDecoder
    # decoder = load_pretrained_decoder(
    #     CFG['decoder_ckpt'], YourDecoder, CFG['decoder_prefix']
    # )
    raise NotImplementedError(
        "decoder_cls를 실제 Decoder 클래스로 교체하세요.\n"
        "  from your_decoder import YourDecoder\n"
        "  decoder = load_pretrained_decoder(\n"
        "      CFG['decoder_ckpt'], YourDecoder, CFG['decoder_prefix']\n"
        "  )\n"
        "교체 후 이 raise 줄을 제거하세요."
    )

    # ── 모델 ──────────────────────────────────────────────────────────────────
    model = build_model(
        dinov2_path = CFG['dinov2_path'],
        decoder     = decoder,
        dec_chs     = CFG['dec_chs'],
        img_size    = CFG['img_size'],
        freeze_dino = CFG['freeze_dino'],
    ).to(device)

    # ── DataLoader ─────────────────────────────────────────────────────────────
    print("DataLoader 구성...")
    train_ds = PatchDataset(
        patch_dir = f"{CFG['patch_dir']}/train",
        augment   = True,   # HorizontalFlip + VerticalFlip
    )
    val_ds = PatchDataset(
        patch_dir = f"{CFG['patch_dir']}/val",
        augment   = False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size      = CFG['batch_size'],
        shuffle         = True,
        num_workers     = CFG['num_workers'],
        pin_memory      = True,
        prefetch_factor = 2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = CFG['batch_size'],
        shuffle     = False,
        num_workers = CFG['num_workers'],
        pin_memory  = True,
    )

    # ── Loss ───────────────────────────────────────────────────────────────────
    chroma_loss = ChromophoreLoss(
        w_supervised = CFG['w_supervised'],
        w_ssim       = CFG['w_ssim'],
        w_recon      = CFG['w_recon'],
        w_ortho      = CFG['w_ortho'],
        w_smooth     = CFG['w_smooth'],
    ).to(device)

    feat_loss = AmbientInvariantLoss(
        loss_type    = CFG['feat_loss_type'],
        temperature  = CFG['feat_temperature'],
        use_patch    = CFG['use_patch_feat'],
        patch_weight = CFG['patch_feat_w'],
    ).to(device)

    # 조명 augmentation transform
    ambient_transform = get_ambient_transform()

    # ── Optimizer / Scheduler ──────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = CFG['lr'],
        weight_decay = CFG['weight_decay'],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs']
    )

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    print("\n학습 시작...")
    print(f"  Feature Loss: {CFG['feat_loss_type']}  "
          f"(w_feat={CFG['w_feat']})")
    print(f"  Patch feat:   {CFG['use_patch_feat']}")

    best_val = float('inf')

    for epoch in range(1, CFG['epochs'] + 1):

        # Chromophore Loss 가중치 단계적 조절
        chroma_weights = get_loss_weights(epoch, CFG['epochs'])
        for k, v in chroma_weights.items():
            setattr(chroma_loss, k, v)

        train_log = train_one_epoch(
            model, train_loader, chroma_loss, feat_loss,
            ambient_transform, optimizer, device, CFG
        )
        val_log = validate(
            model, val_loader, chroma_loss, feat_loss,
            ambient_transform, device, CFG
        )
        scheduler.step()

        print(
            f"Epoch [{epoch:03d}/{CFG['epochs']}]  "
            f"train={train_log['total']:.4f}  "
            f"val={val_log['total']:.4f}  "
            f"sup={val_log['sup']:.4f}  "
            f"recon={val_log['recon']:.4f}  "
            f"feat_cls={val_log['feat_cls']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_log['total'] < best_val:
            best_val  = val_log['total']
            ckpt_path = Path(CFG['checkpoint_dir']) / 'best_model.pth'
            torch.save({
                'epoch'     : epoch,
                'state_dict': model.state_dict(),
                'val_loss'  : val_log,
                'cfg'       : CFG,
            }, ckpt_path)
            print(f"  ✅ Best 저장: {ckpt_path}")

    print("\n학습 완료!")


if __name__ == '__main__':
    main()
