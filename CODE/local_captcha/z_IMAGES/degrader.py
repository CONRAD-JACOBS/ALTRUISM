import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import random

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def degrade_image(
    img,
    blur_radius=0.0,          # 0 = off
    downsample=1.0,           # 1 = off; e.g., 2.0 halves width/height then upsamples
    contrast=1.0,             # <1 reduces contrast (e.g., 0.9)
    brightness=1.0,           # <1 darker (e.g., 0.95)
    noise_std=0.0,            # 0 = off; ~5–20 is moderate
    seed=None
):
    if seed is not None:
        random.seed(seed)

    img = img.convert("RGB")

    # Downsample then upsample (blocky / low-res feel)
    if downsample and downsample > 1.0:
        w, h = img.size
        nw = max(1, int(w / downsample))
        nh = max(1, int(h / downsample))
        img = img.resize((nw, nh), resample=Image.Resampling.BILINEAR)
        img = img.resize((w, h), resample=Image.Resampling.NEAREST)

    # Blur
    if blur_radius and blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Contrast / brightness
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)

    # Noise (simple per-pixel additive noise)
    if noise_std and noise_std > 0.0:
        import numpy as np
        arr = np.array(img).astype("float32")
        noise = np.random.normal(0.0, noise_std, size=arr.shape).astype("float32")
        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype("uint8")
        img = Image.fromarray(arr, mode="RGB")

    return img

def process_folder(
    in_dir,
    out_dir,
    blur_radius=1.2,
    downsample=1.6,
    contrast=0.95,
    brightness=1.0,
    noise_std=8.0,
    jpeg_quality=45,          # lower = more artifacts; 25–60 typical
    overwrite=False
):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not paths:
        raise RuntimeError(f"No images found in {in_dir}")

    for p in paths:
        out_name = p.stem + ".jpg"   # normalize output to jpg for consistent artifacting
        out_path = out_dir / out_name
        if out_path.exists() and not overwrite:
            continue

        with Image.open(p) as img:
            degraded = degrade_image(
                img,
                blur_radius=blur_radius,
                downsample=downsample,
                contrast=contrast,
                brightness=brightness,
                noise_std=noise_std,
            )
            degraded.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)

    print(f"Done. Wrote {len(paths)} images to: {out_dir}")

if __name__ == "__main__":
    # Example usage:
    # process_folder("TARGETS", "TARGETS_DEGRADED", blur_radius=1.4, downsample=1.8, noise_std=10, jpeg_quality=40)
    blur_radius=1.6
    downsample=1.5
    contrast=0.8
    brightness=1.1
    noise_std=30
    jpeg_quality=40

    process_folder(
    in_dir="TARGETS",
    out_dir="TARGETS_DEGRADED",
    blur_radius=blur_radius,
    downsample=downsample,
    contrast=contrast,
    brightness=brightness,
    noise_std=noise_std,
    jpeg_quality=jpeg_quality,
    overwrite=True
    )   
  
    process_folder(
        in_dir="DISTRACTORS",
        out_dir="DISTRACTORS_DEGRADED",
        blur_radius=blur_radius,
        downsample=downsample,
        contrast=contrast,
        brightness=brightness,
        noise_std=noise_std,
        jpeg_quality=jpeg_quality,
        overwrite=True
        )
