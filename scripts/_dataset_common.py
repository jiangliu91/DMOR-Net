
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Common lightweight dataset loaders for DMOR-Edge extra benchmarks (NYUDv2, BIPED).

NYUDv2 (expected one of):
  dataset/NYUDv2/
    RGB/{train|val|test}/*.(png|jpg)
    HHA/{train|val|test}/*.(png|jpg)
    GT/{train|val|test}/*.png

BIPED (expected one of):
  dataset/BIPEDv1/ or dataset/BIPEDv2/
    images/{train|val|test}/*.(png|jpg)
    gt/{train|val|test}/*.png

The loaders match by filename stem.
"""
import os
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    out: List[Path] = []
    for ext in _IMG_EXTS:
        out += sorted(d.glob(f"*{ext}"))
    return out

def _read_rgb(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def _read_gray(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Cannot read GT: {path}")
    return im

def _to_tensor_img(im: np.ndarray) -> torch.Tensor:
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))  # CHW
    return torch.from_numpy(im)

def _to_tensor_gt(gt: np.ndarray) -> torch.Tensor:
    gt = (gt > 127).astype(np.float32)
    return torch.from_numpy(gt).unsqueeze(0)

class PairedEdgeDataset(Dataset):
    """Generic paired dataset: image_dir + gt_dir, match by stem."""
    def __init__(self, image_dir: str, gt_dir: str):
        self.image_dir = Path(image_dir)
        self.gt_dir = Path(gt_dir)
        imgs = _list_images(self.image_dir)
        if not imgs:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        self.items: List[Tuple[Path, Path]] = []
        for p in imgs:
            stem = p.stem
            gt = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                cand = self.gt_dir / f"{stem}{ext}"
                if cand.exists():
                    gt = cand
                    break
            if gt is None:
                # try prefix/suffix variants
                for cand in self.gt_dir.glob(f"{stem}*"):
                    if cand.suffix.lower() in _IMG_EXTS:
                        gt = cand
                        break
            if gt is None:
                continue
            self.items.append((p, gt))
        if not self.items:
            raise FileNotFoundError(
                f"Found {len(imgs)} images in {self.image_dir} but matched 0 gts in {self.gt_dir}. "
                "Make sure filenames align (same stem)."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        ip, gp = self.items[idx]
        im = _read_rgb(ip)
        gt = _read_gray(gp)
        return _to_tensor_img(im), _to_tensor_gt(gt), ip.stem

def guess_nyud_paths(root: str, split: str, modality: str):
    r = Path(root)
    cand_rgb = [r/"RGB"/split, r/"images"/"rgb"/split, r/"images"/split]
    cand_hha = [r/"HHA"/split, r/"hha"/split, r/"images"/"hha"/split]
    cand_gt  = [r/"GT"/split, r/"gt"/split, r/"edge"/split, r/"edges"/split]
    def pick(cands):
        for c in cands:
            if c.exists() and _list_images(c):
                return c
        return None
    modality = modality.lower()
    if modality == "rgb":
        img_dir = pick(cand_rgb)
    elif modality == "hha":
        img_dir = pick(cand_hha)
    else:
        raise ValueError("modality must be rgb or hha (fusion handled elsewhere).")
    gt_dir = pick(cand_gt)
    if img_dir is None or gt_dir is None:
        raise FileNotFoundError(
            f"Cannot auto-detect NYUDv2 paths under {root}. Expected e.g. RGB/{split} and GT/{split}."
        )
    return str(img_dir), str(gt_dir)

def guess_biped_paths(root: str, split: str):
    r = Path(root)
    cand_img = [r/"images"/split, r/"imgs"/split, r/"RGB"/split, r/split/"images"]
    cand_gt  = [r/"gt"/split, r/"GT"/split, r/"edges"/split, r/"edge"/split, r/split/"gt"]
    def pick(cands):
        for c in cands:
            if c.exists() and _list_images(c):
                return c
        return None
    img_dir = pick(cand_img)
    gt_dir  = pick(cand_gt)
    if img_dir is None or gt_dir is None:
        raise FileNotFoundError(
            f"Cannot auto-detect BIPED paths under {root}. Expected e.g. images/{split} and gt/{split}."
        )
    return str(img_dir), str(gt_dir)
