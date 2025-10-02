# batch_infer.py
import os, glob, torch
from PIL import Image
from tqdm import tqdm
from train_hybrid_pytorch import HybridColorizationDataset, HybridColorizationModel, lab_norm_to_rgb

opt_dir, sar_dir = "data/optical", "data/sar"
pairs = []
for p in sorted(glob.glob(os.path.join(opt_dir, "*.png"))):
    stem = os.path.splitext(os.path.basename(p))[0]
    q = os.path.join(sar_dir, stem + ".tif")
    if os.path.exists(q): pairs.append((p,q))
assert pairs, "No pairs found."

# discover physics channel count once
ds0 = HybridColorizationDataset([pairs[0][0]],[pairs[0][1]], resize_hw=(224,224))
L0,_,L3_0,phys0,_ = ds0[0]
model = HybridColorizationModel(phys_ch=phys0.shape[0], codebook_K=16)
model.load_state_dict(torch.load("hybrid_physics_model.pth", map_location="cpu"))
model.eval()

os.makedirs("runs", exist_ok=True)

with torch.no_grad():
    for o,s in tqdm(pairs, desc="Colorizing"):
        ds = HybridColorizationDataset([o],[s], resize_hw=(224,224))
        L,_,L3,phys,_ = ds[0]
        ab,_ = model(L3.unsqueeze(0), phys.unsqueeze(0))
        rgb = lab_norm_to_rgb(L.unsqueeze(0), ab)
        out = os.path.join("runs", os.path.splitext(os.path.basename(o))[0] + "_pred.png")
        Image.fromarray(rgb).save(out)
print("Saved colorized images to runs/")
