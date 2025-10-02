# create_synth.py
import os, numpy as np
from PIL import Image, ImageDraw

os.makedirs("data/optical", exist_ok=True)
os.makedirs("data/sar", exist_ok=True)

for i in range(100):
    H = W = 256

    # Make an RGB image with simple shapes
    rgb_img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(rgb_img)
    d.ellipse((W//2-60, H//2-60, W//2+60, H//2+60), fill=(0,255,0))
    d.rectangle((30,30,90,90), fill=(255,0,0))
    d.line((50,200,200,220), fill=(0,0,255), width=5)

    # Convert to "SAR-like" grayscale + noise
    sar = np.array(rgb_img.convert("L")).astype(np.float32)
    sar += np.random.randn(H, W).astype(np.float32) * 20.0
    sar = np.clip(sar, 0, 255).astype(np.uint8)
    sar_img = Image.fromarray(sar, mode="L")

    name = f"img_{i:03d}"
    rgb_img.save(f"data/optical/{name}.png")            # RGB
    sar_img.save(f"data/sar/{name}.tif", format="TIFF") # SAR (grayscale)

print("Created 10 synthetic pairs in data/optical and data/sar.")
