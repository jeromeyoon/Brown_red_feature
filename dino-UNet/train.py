"""
dino-UNet/train.py
==================
DinoUNet 학습 메인 스크립트

기존 train.py와의 차이
----------------------
ChromophoreNet train.py:
  - 외부 Decoder checkpoint 로드 필요 (decoder_ckpt, decoder_cls)
  - build_model(dinov2_path, decoder, ...)

DinoUNet train.py:
  - Decoder checkpoint 불필요 → end-to-end 학습
  - build_dino_unet(dinov2_path, ...)
  - Phase 1 → Phase 2 자동 전환 (DINOv2 unfreeze)

전체 Loss:
  total = loss_chroma + w_feat * loss_feat

실행:
  python train.py
  nohup python train.py > train_dino_unet.log 2>&1 &
"""

import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 상위 폴더(yoon_work)의 dataset, ambient_aug 재사용
sys.path.append(str(Path(__file__).parent.parent))
from dataset     import PatchDataset
from ambient_aug import get_ambient_transform, apply_ambient_aug

from model import build_dino_unet
from loss  import ChromophoreLoss, AmbientInvariantLoss, get_loss_weights


# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # 경로
    patch_dir      = '../patches',                  # extract_patches.py 출력
    dinov2_path    = '../pretrained/dinov2-base',   # ← 수정
    checkpoint_dir = './checkpoints',

    # 모델
    dec_chs        = [512, 256, 128, 64],
    img_size       = 224,

    # Phase 1: DINOv2 frozen (빠른 수렴)
    # Phase 2: DINOv2 마지막 N 블록 unfreeze (fine-tuning)
    phase2_start_epoch = 30,     # 이 epoch부터 Phase 2 진입
    unfreeze_last      = 4,      # Phase 2에서 unfreeze할 DINOv2 블록 수

    # 학습
    epochs         = 80,
    batch_size     = 16,
    lr             = 1e-4,
    lr_dino        = 1e-5,       # Phase 2 DINOv2 fine-tuning lr
    weight_decay   = 1e-4,
    num_workers    = 8,

    # Feature Consistency Loss
    feat_loss_type  = 'cosine',
    feat_temperature= 0.07,
    use_patch_feat  = False,
    patch_feat_w    = 0.5,
    w_feat          = 0.5,

    # Chromophore Loss 초기 가중치
    w_supervised   = 1.0,
    w_ssim         = 0.0,
    w_recon        = 0.5,
    w_ortho        = 0.0,
    w_smooth       = 0.0,
)


# ══════════════════════════════════════════════════════════════════════════════
# 조명변형 배치 생성
# ══════════════════════════════════════════════════════════════════════════════
def make_ambient_batch(rgb      : torch.Tensor,
                       transform,
                       device   : torch.device) -> torch.Tensor:
    rgb_np  = rgb.cpu().permute(0, 2, 3, 1).numpy()
    aug_np  = apply_ambient_aug(transform, rgb_np)
    rgb_aug = torch.from_numpy(
        aug_np.transpose(0, 3, 1, 2)
    ).float().to(device)
    return rgb_aug


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 전환: Optimizer에 DINOv2 파라미터 추가
# ══════════════════════════════════════════════════════════════════════════════
def enter_phase2(model, optimizer, cfg):
    """
    DINOv2 마지막 N 블록 unfreeze 후 Optimizer에 파라미터 그룹 추가

    기존 optimizer param_groups 유지 + DINOv2 그룹 추가
    → Adapter/Decoder lr과 DINOv2 lr을 분리 관리
    """
    model.unfreeze_dino(last_n_blocks=cfg['unfreeze_last'])

    dino_params = [p for p in model.dinov2.parameters() if p.requires_grad]
    optimizer.add_param_group({
        'params'      : dino_params,
        'lr'          : cfg['lr_dino'],
        'weight_decay': cfg['weight_decay'],
    })
    print(f"  Phase 2 진입: DINOv2 마지막 {cfg['unfreeze_last']}블록 unfreeze")
    print(f"  DINOv2 lr = {cfg['lr_dino']:.1e}  |  "
          f"Adapter/Decoder lr = {cfg['lr']:.1e}")


# ══════════════════════════════════════════════════════════════════════════════
# 학습 / 검증
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, chroma_loss, feat_loss,
                    ambient_transform, optimizer, device, cfg):
    model.train()
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

        with torch.no_grad():
            _, _, cls_aug, patch_aug = model(rgb_aug, return_feat=True)

        loss_c, detail_c = chroma_loss(
            mel_pred   = mel_pred,
            hem_pred   = hem_pred,
            mel_gt     = mel_gt,
            hem_gt     = hem_gt,
            rgb_linear = rgb,
            face_mask  = face_mask,
        )
        loss_f, detail_f = feat_loss(
            cls_orig   = cls_orig,
            cls_aug    = cls_aug,
            patch_orig = patch_orig if cfg['use_patch_feat'] else None,
            patch_aug  = patch_aug  if cfg['use_patch_feat'] else None,
        )

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
    return {
        'total': total_loss / n,
        **{k: v / n for k, v in chroma_log.items()},
        **{k: v / n for k, v in feat_log.items()},
    }


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
    return {
        'total': total_loss / n,
        **{k: v / n for k, v in chroma_log.items()},
        **{k: v / n for k, v in feat_log.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    Path(CFG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # ── 모델 (외부 Decoder 불필요) ─────────────────────────────────────────────
    model = build_dino_unet(
        dinov2_path  = CFG['dinov2_path'],
        dec_chs      = CFG['dec_chs'],
        img_size     = CFG['img_size'],
        freeze_dino  = True,   # Phase 1: DINOv2 frozen
    ).to(device)

    # ── DataLoader ─────────────────────────────────────────────────────────────
    print("DataLoader 구성...")
    train_ds = PatchDataset(
        patch_dir = f"{CFG['patch_dir']}/train",
        augment   = True,
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

    ambient_transform = get_ambient_transform()

    # ── Optimizer (Phase 1: Adapter + Decoder만) ───────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = CFG['lr'],
        weight_decay = CFG['weight_decay'],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs']
    )

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    print(f"\n학습 시작 (총 {CFG['epochs']} epoch)")
    print(f"  Phase 1: epoch 1 ~ {CFG['phase2_start_epoch']-1}  "
          f"(DINOv2 frozen)")
    print(f"  Phase 2: epoch {CFG['phase2_start_epoch']} ~ {CFG['epochs']}  "
          f"(DINOv2 마지막 {CFG['unfreeze_last']}블록 unfreeze)")

    best_val   = float('inf')
    in_phase2  = False

    for epoch in range(1, CFG['epochs'] + 1):

        # Phase 2 전환
        if epoch == CFG['phase2_start_epoch'] and not in_phase2:
            enter_phase2(model, optimizer, CFG)
            in_phase2 = True

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

        phase_tag = 'P2' if in_phase2 else 'P1'
        print(
            f"[{phase_tag}] Epoch [{epoch:03d}/{CFG['epochs']}]  "
            f"train={train_log['total']:.4f}  "
            f"val={val_log['total']:.4f}  "
            f"sup={val_log['sup']:.4f}  "
            f"recon={val_log['recon']:.4f}  "
            f"feat_cls={val_log['feat_cls']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_log['total'] < best_val:
            best_val  = val_log['total']
            ckpt_path = Path(CFG['checkpoint_dir']) / 'best_dino_unet.pth'
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
