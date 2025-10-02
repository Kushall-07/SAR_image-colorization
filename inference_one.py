# inference_one.py
import torch
from PIL import Image
from train_hybrid_pytorch import (
    HybridColorizationDataset, HybridColorizationModel, lab_norm_to_rgb
)

optical_path = "data/optical/img_000.png"
sar_path     = "data/sar/img_000.tif"
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = HybridColorizationDataset([optical_path],[sar_path], resize_hw=(img_size, img_size))
L, ab_true, L3, phys, mask = ds[0]
phys_ch = phys.shape[0]

model = HybridColorizationModel(phys_ch=phys_ch, codebook_K=16).to(device)
model.load_state_dict(torch.load("hybrid_physics_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    ab_pred, _ = model(L3.unsqueeze(0).to(device), phys.unsqueeze(0).to(device))
    rgb = lab_norm_to_rgb(L.unsqueeze(0).to(device), ab_pred.to(device))

Image.fromarray(rgb).save("out_one.png")
print("Saved -> out_one.png")
