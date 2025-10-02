# train_hybrid_pytorch.py
# ------------------------------------------------------------
# Hybrid Physics-Guided Colorization (PyTorch-only)
#  - Dataset + physics features (SAR/LIA/VV/VH)
#  - ResNet50 + DenseNet121 encoder with FiLM modulation
#  - U-Net style decoder → ab + log-variance (heteroscedastic)
#  - Loss = NLL + palette regularizer + physics suppression
#  - Training + validation + test (SSIM/PSNR)
# ------------------------------------------------------------
# Deps (pip):
#   torch torchvision torchaudio
#   scikit-image opencv-python pillow matplotlib tqdm
# ------------------------------------------------------------

import os
import math
import argparse
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights

# =========================
# Color space helpers
# =========================
def rgb_to_lab_tensors(img_rgb_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """HxWx3 RGB (uint8/float) -> L:[1,H,W] in [-1,1], ab:[2,H,W] in [-1,1]."""
    from skimage.color import rgb2lab
    if img_rgb_np.dtype != np.float32 and img_rgb_np.dtype != np.float64:
        img_rgb_np = img_rgb_np.astype(np.float32) / 255.0
    lab = rgb2lab(img_rgb_np).astype("float32")  # L in [0,100], a,b ~ [-128,127]
    lab_t = transforms.ToTensor()(lab)            # [3,H,W]
    L = lab_t[[0], ...]
    ab = lab_t[[1, 2], ...]
    # normalize
    L = 2 * (L - 0.0) / 100.0 - 1.0
    ab = 2 * (ab - (-128.0)) / (127.0 - (-128.0)) - 1.0
    return L, ab


def lab_norm_to_rgb(L_norm: torch.Tensor, ab_norm: torch.Tensor) -> np.ndarray:
    """(L,ab) normalized tensors -> RGB uint8 (HxWx3). Uses the first sample in batch."""
    from skimage.color import lab2rgb
    L = (L_norm + 1.0) * 0.5 * 100.0
    a = (ab_norm[:, [0], ...] + 1.0) * 0.5 * (127.0 - (-128.0)) + (-128.0)
    b = (ab_norm[:, [1], ...] + 1.0) * 0.5 * (127.0 - (-128.0)) + (-128.0)
    Lab = torch.cat([L, a, b], dim=1)  # [N,3,H,W]
    Lab_np = Lab[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    rgb = lab2rgb(Lab_np)  # float [0,1]
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
    return rgb_uint8

# =========================
# Physics features (edit for your stack)
# =========================
class PhysicsFeatureExtractor:
    """Build physics channels from SAR and ancillary rasters.
       Channels (example): log(SAR), grad_mag, speckle_var, sin(LIA), cos(LIA), VV/VH ratio.
    """
    def __init__(self, use_grad=True, use_speckle_var=True):
        self.use_grad = use_grad
        self.use_speckle_var = use_speckle_var

    def __call__(self, sar: np.ndarray, lia: Optional[np.ndarray] = None,
                 vv: Optional[np.ndarray] = None, vh: Optional[np.ndarray] = None) -> np.ndarray:
        H, W = sar.shape
        chs: List[np.ndarray] = []
        sar = sar.astype(np.float32)
        # robust clip
        sar = np.clip(sar, np.percentile(sar, 1), np.percentile(sar, 99))
        # log/standardize
        sar_log = np.log1p(np.maximum(sar, 0))
        sar_log = (sar_log - sar_log.mean()) / (sar_log.std() + 1e-6)
        chs.append(sar_log)
        # gradient magnitude
        if self.use_grad:
            gx = cv2.Sobel(sar, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(sar, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx * gx + gy * gy)
            grad_mag = (grad_mag - grad_mag.mean()) / (grad_mag.std() + 1e-6)
            chs.append(grad_mag)
        # local variance (speckle-ish)
        if self.use_speckle_var:
            k = 7
            mu = cv2.blur(sar, (k, k))
            mu2 = cv2.blur(sar * sar, (k, k))
            var = np.clip(mu2 - mu * mu, 0, None)
            var = (var - var.mean()) / (var.std() + 1e-6)
            chs.append(var)
        # LIA
        if lia is not None:
            lia = lia.astype(np.float32)
            if lia.mean() > 3.2:  # likely degrees
                lia = np.deg2rad(lia)
            chs.append(np.sin(lia))
            chs.append(np.cos(lia))
        # VV/VH ratio
        if vv is not None and vh is not None:
            ratio = vv / (vh + 1e-6)
            ratio = np.log1p(np.abs(ratio)) * np.sign(ratio)
            ratio = (ratio - ratio.mean()) / (ratio.std() + 1e-6)
            chs.append(ratio)
        physics = np.stack(chs, axis=0)  # [C,H,W]
        return physics.astype(np.float32)

# =========================
# Dataset
# =========================
class HybridColorizationDataset(Dataset):
    def __init__(self,
                 rgb_paths: List[str],
                 sar_paths: List[str],
                 lia_paths: Optional[List[str]] = None,
                 vv_paths: Optional[List[str]] = None,
                 vh_paths: Optional[List[str]] = None,
                 mask_paths: Optional[List[str]] = None,
                 resize_hw: Tuple[int, int] = (224, 224)):
        assert len(rgb_paths) == len(sar_paths), "rgb_paths and sar_paths must have same length"
        self.rgb_paths = rgb_paths
        self.sar_paths = sar_paths
        self.lia_paths = lia_paths
        self.vv_paths  = vv_paths
        self.vh_paths  = vh_paths
        self.mask_paths = mask_paths
        self.H, self.W = resize_hw
        self.physx = PhysicsFeatureExtractor()

    def __len__(self):
        return len(self.rgb_paths)

    def _read_gray(self, path: str) -> np.ndarray:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(path)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return im.astype(np.float32)

    def __getitem__(self, idx):
        # optical RGB → L, ab (normalized)
        rgb = np.array(Image.open(self.rgb_paths[idx]).convert("RGB"))
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
        L, ab = rgb_to_lab_tensors(rgb)  # [1,H,W], [2,H,W]

        # physics rasters
        sar = self._read_gray(self.sar_paths[idx])
        lia = self._read_gray(self.lia_paths[idx]) if self.lia_paths else None
        vv  = self._read_gray(self.vv_paths[idx])  if self.vv_paths  else None
        vh  = self._read_gray(self.vh_paths[idx])  if self.vh_paths  else None

        physics_np = self.physx(sar, lia, vv, vh)  # [C,H,W]
        physics = torch.from_numpy(physics_np)

        # optional mask (e.g., SAR shadow/layover) where chroma should be suppressed
        if self.mask_paths:
            m = self._read_gray(self.mask_paths[idx])
            m = (m > 0.5 * m.max()).astype(np.float32)
        else:
            m = np.zeros((self.H, self.W), dtype=np.float32)
        mask_t = torch.from_numpy(m)[None, ...]

        # replicate L→L3 for 3-channel encoders
        L3 = L.repeat(1, 3, 1, 1)[0]  # [3,H,W]

        return L, ab, L3, physics, mask_t

# =========================
# FiLM (physics modulation)
# =========================
class FiLM(nn.Module):
    def __init__(self, phys_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(phys_ch, max(32, phys_ch), 3, padding=1),
            nn.BatchNorm2d(max(32, phys_ch)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(32, phys_ch), 2 * out_ch, 1),
        )

    def forward(self, feat: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        phys_rs = F.interpolate(phys, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        gamma_beta = self.block(phys_rs)
        C = feat.shape[1]
        gamma, beta = torch.split(gamma_beta, [C, C], dim=1)
        return feat * (1 + torch.tanh(gamma)) + beta

# =========================
# Encoder (ResNet50 + DenseNet121) + fusion
# =========================
class EnsembleEncoderHybrid(nn.Module):
    def __init__(self, phys_ch: int):
        super().__init__()
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.densenet121 = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])  # up to layer4
        self.densenet121.classifier = nn.Identity()
        self.dense_feats = self.densenet121.features

        self.conv1x1_res = nn.ModuleList([
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(1024, 512, 1),
            nn.Conv2d(2048, 1024, 1),
        ])
        self.conv1x1_den = nn.ModuleList([
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(1024, 512, 1),
            nn.Conv2d(1024, 1024, 1),
        ])

        self.film_res = nn.ModuleList([
            FiLM(phys_ch, 128), FiLM(phys_ch, 256), FiLM(phys_ch, 512), FiLM(phys_ch, 1024)
        ])
        self.film_den = nn.ModuleList([
            FiLM(phys_ch, 128), FiLM(phys_ch, 256), FiLM(phys_ch, 512), FiLM(phys_ch, 1024)
        ])

        self.fusion = nn.ModuleList([
            self._fusion_block(128, 128),
            self._fusion_block(256, 256),
            self._fusion_block(512, 512),
            self._fusion_block(1024, 1024),
        ])

    def _fusion_block(self, c_res: int, c_den: int):
        return nn.Sequential(nn.Conv2d(c_res + c_den, c_res, 1), nn.BatchNorm2d(c_res), nn.ReLU(inplace=True))

    def forward(self, L3: torch.Tensor, phys: torch.Tensor) -> List[torch.Tensor]:
        # ResNet features: indices 4..7 -> [56,28,14,7]
        r = []
        x = L3
        for i, layer in enumerate(self.resnet50.children()):
            x = layer(x)
            if i in [4, 5, 6, 7]:
                r.append(self.conv1x1_res[i - 4](x))  # [128,256,512,1024]
        # DenseNet features: [4,6,8,11]
        d = []
        z = L3
        idx = 0
        for i, layer in enumerate(self.dense_feats):
            z = layer(z)
            if i in [4, 6, 8, 11]:
                d.append(self.conv1x1_den[idx](z))
                idx += 1
        fused = []
        for s in range(4):
            r_m = self.film_res[s](r[s], phys)
            d_m = self.film_den[s](d[s], phys)
            f = torch.cat([r_m, d_m], dim=1)
            f = self.fusion[s](f)
            fused.append(f)
        # order: [56x56, 28x28, 14x14, 7x7]
        return fused

# =========================
# Decoder (uncertainty) → ab + logvar
# =========================
class DecoderUnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 7 -> 14
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 14 -> 28
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 28 -> 56
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 56 -> 112
        )
        # Final conv + upsample to reach 224x224
        self.out = nn.Sequential(
            nn.Conv2d(64, 4, 3, padding=1),  # [a,b, logvar_a, logvar_b]
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 112 -> 224
        )

    def forward(self, f7, f14, f28, f56):
        x = self.dec1(f7)
        x = torch.cat([x, f14], dim=1)
        x = self.dec2(x)
        x = torch.cat([x, f28], dim=1)
        x = self.dec3(x)
        x = torch.cat([x, f56], dim=1)
        x = self.dec4(x)
        y = self.out(x)   # [N,4,224,224]
        ab = torch.tanh(y[:, 0:2, ...])
        logvar = y[:, 2:4, ...]
        return ab, logvar

# =========================
# Full model + palette reg
# =========================
class HybridColorizationModel(nn.Module):
    def __init__(self, phys_ch: int, codebook_K: int = 16):
        super().__init__()
        self.encoder = EnsembleEncoderHybrid(phys_ch=phys_ch)
        self.decoder = DecoderUnc()
        self.codebook = nn.Parameter(torch.rand(codebook_K, 2) * 2 - 1)
        self.tau = 5.0

    def forward(self, L3: torch.Tensor, phys: torch.Tensor):
        f56, f28, f14, f7 = self.encoder(L3, phys)
        ab, logvar = self.decoder(f7, f14, f28, f56)
        return ab, logvar

    def palette_loss(self, ab_pred: torch.Tensor) -> torch.Tensor:
        N, _, H, W = ab_pred.shape
        ab = ab_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        cb = self.codebook[None, :, :].expand(ab.shape[0], -1, -1)
        d2 = torch.sum((ab[:, None, :] - cb) ** 2, dim=-1)
        softmin = -torch.logsumexp(-self.tau * d2, dim=1) / self.tau
        return softmin.mean()

# =========================
# Losses
# =========================
def heteroscedastic_nll(ab_pred: torch.Tensor, logvar: torch.Tensor, ab_true: torch.Tensor) -> torch.Tensor:
    err2 = (ab_pred - ab_true) ** 2
    nll = 0.5 * (torch.exp(-logvar) * err2 + logvar)
    return nll.mean()

def physics_chroma_suppression(ab_pred: torch.Tensor, phys_mask: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    chroma = torch.sqrt(torch.clamp(ab_pred[:, 0:1, ...] ** 2 + ab_pred[:, 1:2, ...] ** 2, 1e-8))
    return weight * (chroma * phys_mask).mean()

# =========================
# Train / Val loop
# =========================
@torch.no_grad()
def evaluate_rgb_metrics(model: HybridColorizationModel, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    model.eval()
    tot_ssim = 0.0; tot_psnr = 0.0; N = 0
    for L, ab, L3, phys, mask in loader:
        L, ab, L3, phys = L.to(device), ab.to(device), L3.to(device), phys.to(device)
        ab_pred, _ = model(L3, phys)
        rgb_true = lab_norm_to_rgb(L, ab) / 255.0
        rgb_pred = lab_norm_to_rgb(L, ab_pred) / 255.0
        s = ssim(rgb_true, rgb_pred, channel_axis=2, data_range=1.0)
        p = psnr(rgb_true, rgb_pred, data_range=1.0)
        tot_ssim += float(s); tot_psnr += float(p); N += 1
    return tot_ssim / max(1, N), tot_psnr / max(1, N)

def train(model: HybridColorizationModel,
          train_loader: DataLoader,
          val_loader: DataLoader,
          device: torch.device,
          epochs: int = 30,
          lr: float = 1e-3,
          w_palette: float = 0.05,
          w_phys: float = 0.02,
          save_path: str = "hybrid_physics_model.pth"):
    # (Optional) freeze heavy backbone layers to save memory
    try:
        for m in model.encoder.modules():
            if isinstance(m, (models.resnet.Bottleneck, models.densenet._DenseLayer, models.densenet._Transition)):
                for p in m.parameters():
                    p.requires_grad = False
    except Exception:
        pass

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=3)

    best_val = float('inf')
    for ep in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}")
        for L, ab, L3, phys, mask in pbar:
            L, ab, L3, phys, mask = L.to(device), ab.to(device), L3.to(device), phys.to(device), mask.to(device)
            optim.zero_grad()
            ab_pred, logvar = model(L3, phys)
            loss_nll  = heteroscedastic_nll(ab_pred, logvar, ab)
            loss_pal  = model.palette_loss(ab_pred)
            loss_phys = physics_chroma_suppression(ab_pred, mask, weight=1.0)
            loss = loss_nll + w_palette * loss_pal + w_phys * loss_phys
            loss.backward()
            optim.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{running/(pbar.n or 1):.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L, ab, L3, phys, mask in val_loader:
                L, ab, L3, phys, mask = L.to(device), ab.to(device), L3.to(device), phys.to(device), mask.to(device)
                ab_pred, logvar = model(L3, phys)
                l_nll  = heteroscedastic_nll(ab_pred, logvar, ab)
                l_pal  = model.palette_loss(ab_pred)
                l_phys = physics_chroma_suppression(ab_pred, mask, weight=1.0)
                val_loss += (l_nll + w_palette * l_pal + w_phys * l_phys).item()
        val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {ep+1}/{epochs} | train {running/len(train_loader):.4f} | val {val_loss:.4f}")
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model → {save_path} (val {best_val:.4f})")

# =========================
# CLI / Main
# =========================
def main():
    ap = argparse.ArgumentParser("Hybrid Physics-Guided Colorization (PyTorch)")
    ap.add_argument("--optical_dir", type=str, required=True, help="Folder of RGB images (png/jpg)")
    ap.add_argument("--sar_dir",     type=str, required=True, help="Folder of SAR rasters (tif)")
    ap.add_argument("--lia_dir",     type=str, default=None)
    ap.add_argument("--vv_dir",      type=str, default=None)
    ap.add_argument("--vh_dir",      type=str, default=None)
    ap.add_argument("--mask_dir",    type=str, default=None)
    ap.add_argument("--ext_rgb",     type=str, default="png", help="optical extension (png|jpg|jpeg)")
    ap.add_argument("--img_size",    type=int, default=224)
    ap.add_argument("--batch",       type=int, default=8)
    ap.add_argument("--epochs",      type=int, default=30)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--w_palette",   type=float, default=0.05)
    ap.add_argument("--w_phys",      type=float, default=0.02)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # discover files & align paths by basename
    rgb_paths = sorted(glob(os.path.join(args.optical_dir, f"*.{args.ext_rgb}")))
    assert len(rgb_paths) > 0, f"No optical images found in {args.optical_dir}"

    def map_like(rgb_p: str, new_root: Optional[str], new_ext: str) -> Optional[str]:
        if not new_root: return None
        base = os.path.splitext(os.path.basename(rgb_p))[0]
        cand = os.path.join(new_root, f"{base}.{new_ext}")
        return cand if os.path.isfile(cand) else None

    sar_paths  = []
    lia_paths  = [] if args.lia_dir else None
    vv_paths   = [] if args.vv_dir else None
    vh_paths   = [] if args.vh_dir else None
    mask_paths = [] if args.mask_dir else None

    for p in rgb_paths:
        sar_p = map_like(p, args.sar_dir, "tif")
        if sar_p is None:
            raise FileNotFoundError(f"Matching SAR .tif not found for {p}")
        sar_paths.append(sar_p)
        if lia_paths is not None:  lia_paths.append(map_like(p, args.lia_dir, "tif"))
        if vv_paths  is not None:  vv_paths.append(map_like(p, args.vv_dir,  "tif"))
        if vh_paths  is not None:  vh_paths.append(map_like(p, args.vh_dir,  "tif"))
        if mask_paths is not None: mask_paths.append(map_like(p, args.mask_dir,"tif"))

    H = W = args.img_size
    ds = HybridColorizationDataset(
        rgb_paths,
        sar_paths,
        lia_paths if args.lia_dir else None,
        vv_paths if args.vv_dir else None,
        vh_paths if args.vh_dir else None,
        mask_paths if args.mask_dir else None,
        resize_hw=(H, W)
    )

    n = len(ds)
    tr_n = int(0.8 * n); va_n = int(0.1 * n); te_n = n - tr_n - va_n
    tr_set, va_set, te_set = random_split(ds, [tr_n, va_n, te_n])

    train_loader = DataLoader(tr_set, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, drop_last=True)
    val_loader   = DataLoader(va_set,  batch_size=args.batch, shuffle=False, num_workers=args.num_workers, drop_last=False)  # <- changed to False
    test_loader  = DataLoader(te_set,  batch_size=args.batch, shuffle=False, num_workers=args.num_workers, drop_last=False)


    device = torch.device(args.device)
    # peek to get physics channel count
    L, ab, L3, phys, mask = ds[0]
    phys_ch = phys.shape[0]
    model = HybridColorizationModel(phys_ch=phys_ch, codebook_K=16).to(device)

    train(model, train_loader, val_loader, device,
          epochs=args.epochs, lr=args.lr,
          w_palette=args.w_palette, w_phys=args.w_phys,
          save_path="hybrid_physics_model.pth")

    ssim_v, psnr_v = evaluate_rgb_metrics(model, test_loader, device)
    print(f"Test SSIM: {ssim_v:.4f} | PSNR: {psnr_v:.2f} dB")

if __name__ == "__main__":
    main()
