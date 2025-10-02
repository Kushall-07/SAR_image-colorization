# colorize_gray.py
import sys, torch, numpy as np
from PIL import Image
from skimage.color import rgb2lab
from train_hybrid_pytorch import HybridColorizationDataset, HybridColorizationModel, lab_norm_to_rgb

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the path you pass on the command line; else fall back to this default
gray_path = sys.argv[1] if len(sys.argv) > 1 else "data/gray_inputs/my_gray.png"

# 1) Discover physics channel count from any training pair you already have
probe_opt = "data/optical/img_000.png"
probe_sar = "data/sar/img_000.tif"
ds0 = HybridColorizationDataset([probe_opt],[probe_sar], resize_hw=(IMG_SIZE, IMG_SIZE))
_, _, _, phys0, _ = ds0[0]
phys_ch = phys0.shape[0]

# 2) Load grayscale → build L ([-1,1]) and 3-channel L input
g = Image.open(gray_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
g3 = np.stack([np.array(g)]*3, axis=2)                      # HxWx3
Lab = rgb2lab(g3).astype(np.float32)                         # L:[0,100]
L = Lab[..., 0]
L_norm = 2.0*(L/100.0) - 1.0                                 # [-1,1]
L_t = torch.from_numpy(L_norm)[None, ...].float()            # [1,H,W]
L3_t = L_t.repeat(3,1,1)                                     # [3,H,W]

# 3) Zero physics (we’re not providing SAR for gray-only inference)
phys = torch.zeros(phys_ch, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

# 4) Load your trained model and run inference
model = HybridColorizationModel(phys_ch=phys_ch, codebook_K=16).to(device)
model.load_state_dict(torch.load("hybrid_physics_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    ab_pred, _ = model(L3_t.unsqueeze(0).to(device), phys.unsqueeze(0).to(device))  # [1,2,H,W]
    rgb = lab_norm_to_rgb(L_t.unsqueeze(0).to(device), ab_pred.to(device))           # uint8 HxWx3

out_path = "out_gray_only.png"
Image.fromarray(rgb).save(out_path)
print(f"Saved -> {out_path}")
