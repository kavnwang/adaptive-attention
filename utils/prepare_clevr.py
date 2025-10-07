#!/usr/bin/env python3
"""
Download a small subset of CLEVR and generate a summaries JSON compatible with
train_visual_compressor.py.

This script downloads:
  - Scenes JSON for the chosen split (train/val/test)
  - A limited number of images from the same split
It then creates an annotations JSON with entries of the form:
  { "image": "images/<split>/<filename>", "summaries": ["<auto summary>"] }

Usage examples:
  # Attempt download; if it fails (e.g., no network), fallback to synthetic
  python utils/prepare_clevr.py --out_dir data/clevr-mini --split val --num 200 --synthetic_on_fail

  # Force synthetic generation only (no network attempts)
  python utils/prepare_clevr.py --out_dir data/clevr-mini --num 200 --synthetic_only

Note: CLEVR is hosted at Facebook AI public files. This script only fetches a
small subset; the full dataset is large (>15GB).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

try:
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError, URLError
except ImportError:
    print("Python stdlib urllib is required.", file=sys.stderr)
    raise


BASES = [
    "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0",
    "https://cs.stanford.edu/people/jcjohns/clevr/CLEVR_v1.0",
]


def http_download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)


def generate_synthetic_images(out_dir: Path, num: int, image_size: int = 224) -> List[Dict]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise RuntimeError("Pillow (PIL) is required for synthetic generation. Install 'pillow' or avoid synthetic mode.") from e
    images_dir = out_dir / "images" / "synthetic"
    images_dir.mkdir(parents=True, exist_ok=True)

    colors = [(220,20,60), (25,25,112), (34,139,34), (255,140,0), (72,61,139), (0,139,139)]
    shapes = ["circle", "square", "triangle"]
    materials = ["rubber", "metal"]
    sizes = ["small", "large"]

    annotations: List[Dict] = []
    for i in range(num):
        img = Image.new("RGB", (image_size, image_size), (245, 245, 245))
        draw = ImageDraw.Draw(img)

        n_obj = max(1, (i % 6) + 1)
        parts: List[str] = []
        for j in range(n_obj):
            color = colors[(i * 3 + j) % len(colors)]
            shape = shapes[(i + j) % len(shapes)]
            material = materials[(i + 2*j) % len(materials)]
            size = sizes[(i + j) % len(sizes)]
            cx = 20 + ((i * 37 + j * 53) % (image_size - 40))
            cy = 20 + ((i * 51 + j * 29) % (image_size - 40))
            r = 12 if size == "small" else 20
            if shape == "circle":
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
            elif shape == "square":
                draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color)
            else:  # triangle
                draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
            parts.append(f"{size} {material} {shape}")

        filename = f"synthetic_{i:06d}.png"
        img_path = images_dir / filename
        img.save(img_path)

        summary = f"A synthetic CLEVR-like scene with {n_obj} objects: " + "; ".join(parts) + "."
        annotations.append({
            "image": str(Path("images") / "synthetic" / filename),
            "summaries": [summary],
        })

    return annotations


def generate_summary(scene: Dict) -> str:
    objs = scene.get("objects", [])
    parts: List[str] = []
    for o in objs[:12]:  # cap to keep summaries manageable
        color = o.get("color", "")
        size = o.get("size", "")
        material = o.get("material", "")
        shape = o.get("shape", "")
        parts.append(" ".join(p for p in [size, color, material, shape] if p))
    n = len(objs)
    if parts:
        desc = "; ".join(parts)
        return f"A CLEVR scene with {n} objects: {desc}."
    return f"A CLEVR scene with {n} objects."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/clevr-mini")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--num", type=int, default=200, help="How many images to download")
    ap.add_argument("--synthetic_on_fail", action="store_true", help="Generate synthetic images if downloads fail")
    ap.add_argument("--synthetic_only", action="store_true", help="Skip downloads and generate synthetic images only")
    ap.add_argument("--image_size", type=int, default=224, help="Synthetic image size (px)")
    ap.add_argument("--zip_path", type=str, default=None,
                    help="Path to a local CLEVR_v1.0.zip to extract from (offline)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images" / args.split
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    annotations: List[Dict]
    zip_path: Optional[Path] = None
    if args.synthetic_only:
        print("Generating synthetic dataset only (no downloads)...")
        annotations = generate_synthetic_images(out_dir, args.num, image_size=args.image_size)
    else:
        # If user provided a local zip, use it preferentially
        if args.zip_path is not None:
            zp = Path(args.zip_path)
            if zp.exists():
                zip_path = zp
                print(f"Using local CLEVR zip: {zip_path}")
            else:
                print(f"Provided zip_path not found: {zp}")
        # Download scenes JSON (try mirrors)
        scenes_name = f"CLEVR_{args.split}_scenes.json"
        scenes_path = scenes_dir / scenes_name
        try:
            if not scenes_path.exists():
                # First, try extracting from local zip if provided
                if zip_path is not None and zip_path.exists():
                    inner_path = f"CLEVR_v1.0/scenes/{scenes_name}"
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            with zf.open(inner_path) as src, open(scenes_path, 'wb') as dst:
                                dst.write(src.read())
                        print(f"Extracted scenes JSON from zip to {scenes_path}")
                    except KeyError:
                        print("Scenes JSON not found inside provided zip; will attempt mirrors if allowed...")
                # If still missing, attempt network mirrors
                if not scenes_path.exists():
                    last_err = None
                    for base in BASES:
                        scenes_url = f"{base}/scenes/{scenes_name}"
                        print(f"Downloading scenes: {scenes_url}")
                        try:
                            http_download(scenes_url, scenes_path)
                            last_err = None
                            break
                        except (HTTPError, URLError) as e:
                            print(f"  Failed: {e}")
                            last_err = e
                    if last_err is not None:
                        # Fallback: try to download the full CLEVR zip and extract the scenes JSON
                        zip_candidates = []
                        for base in BASES:
                            zip_candidates.append(f"{base}/CLEVR_v1.0.zip")
                            if base.endswith("/CLEVR_v1.0"):
                                zip_candidates.append(base.rsplit("/", 1)[0] + "/CLEVR_v1.0.zip")
                        for zip_url in zip_candidates:
                            try:
                                print(f"Attempting zip download: {zip_url}")
                                zip_path = out_dir / "CLEVR_v1.0.zip"
                                http_download(zip_url, zip_path)
                                break
                            except (HTTPError, URLError) as e2:
                                print(f"  Zip failed: {e2}")
                                zip_path = None
                        if zip_path is None or not zip_path.exists():
                            raise last_err
                        # Extract scenes JSON from zip
                        inner_path = f"CLEVR_v1.0/scenes/{scenes_name}"
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            with zf.open(inner_path) as src, open(scenes_path, 'wb') as dst:
                                dst.write(src.read())
            else:
                print(f"Scenes already present: {scenes_path}")
            with open(scenes_path, "r") as f:
                scenes = json.load(f)
            scenes_list = scenes.get("scenes", [])
            if not scenes_list:
                raise RuntimeError("No scenes found in scenes JSON.")

            subset = scenes_list[: max(0, args.num)]

            # Download/extract images and build annotations
            annotations = []
            total = len(subset)

            # If a local zip is available, open it once to speed up repeated access
            zf = None
            if zip_path is not None and zip_path.exists():
                try:
                    zf = zipfile.ZipFile(zip_path, 'r')
                except Exception:
                    zf = None

            for idx, s in enumerate(subset, 1):
                image_filename = s.get("image_filename")
                if not image_filename:
                    continue
                # Try mirrors
                local_path = images_dir / image_filename
                if not local_path.exists():
                    # First try to extract from zip if available (fast, offline)
                    extracted = False
                    if zf is not None:
                        inner_img = f"CLEVR_v1.0/images/{args.split}/{image_filename}"
                        try:
                            with zf.open(inner_img) as src:
                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(local_path, 'wb') as dst:
                                    dst.write(src.read())
                                    extracted = True
                            # Progress hint every 200 images to avoid looking stuck
                            if idx % 200 == 0:
                                print(f"Extracted {idx}/{total}: {image_filename}")
                        except KeyError:
                            extracted = False
                    if not extracted:
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        last_err = None
                        for base in BASES:
                            image_url = f"{base}/images/{args.split}/{image_filename}"
                            print(f"Downloading image: {image_url}")
                            try:
                                http_download(image_url, local_path)
                                last_err = None
                                break
                            except (HTTPError, URLError) as e:
                                print(f"  Failed: {e}")
                                last_err = e
                        if last_err is not None:
                            raise last_err
                else:
                    print(f"Image already present: {local_path}")
                summary = generate_summary(s)
                annotations.append({
                    "image": str(Path("images") / args.split / image_filename),
                    "summaries": [summary],
                })
            # Ensure we close the zipfile if we opened it
            if zf is not None:
                try:
                    zf.close()
                except Exception:
                    pass

        except (HTTPError, URLError, RuntimeError) as e:
            if args.synthetic_on_fail:
                print(f"Download failed ({e}); generating synthetic dataset instead...")
                annotations = generate_synthetic_images(out_dir, args.num, image_size=args.image_size)
            else:
                print("Download failed and synthetic_on_fail is False.", file=sys.stderr)
                raise

    out_json = out_dir / "clevr_summaries.json"
    with open(out_json, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Wrote {out_json} with {len(annotations)} entries.")


if __name__ == "__main__":
    main()
