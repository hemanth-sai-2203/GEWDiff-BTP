
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import tifffile
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy.ndimage import zoom

from huggingface_hub import HfApi, hf_hub_url

from model.RWT import rwa


HF_REPO = "zhu-xlab/GEWDiff_training_dataset"

OUT_SIZE = 256
COMPACT_BANDS = 121
PCA_BANDS = 20


def build_manifest(output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(
        repo_id=HF_REPO,
        repo_type="dataset",
    )

    gt = sorted(
        f for f in files
        if f.startswith("train/gt/") and f.endswith(".tif")
    )

    masks = {
        Path(f).name
        for f in files
        if f.startswith("train/mask/") and f.endswith(".npy")
    }

    edges = {
        Path(f).name
        for f in files
        if f.startswith("train/edge/") and f.endswith(".npy")
    }

    records = []

    for path in gt:
        stem = Path(path).stem
        mask_name = stem + ".npy"

        records.append({
            "gt": path,
            "mask": (
                f"train/mask/{mask_name}"
                if mask_name in masks else None
            ),
            "edge": (
                f"train/edge/{mask_name}"
                if mask_name in edges else None
            ),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f)

    print("Manifest:", output_path)
    print("GT samples:", len(records))
    print("Masks:", sum(x["mask"] is not None for x in records))
    print("Edges:", sum(x["edge"] is not None for x in records))

    return records


def load_manifest(path: str):
    path = Path(path)

    if not path.exists():
        return build_manifest(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_file(repo_path: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = hf_hub_url(
        repo_id=HF_REPO,
        filename=repo_path,
        repo_type="dataset",
    )

    response = requests.get(
        url,
        stream=True,
        timeout=300,
    )
    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024 * 8):
            if chunk:
                f.write(chunk)

    return destination


def wavelet_level(bands: int) -> int:
    if COMPACT_BANDS - 1 >= int(bands / 2):
        return 1
    elif COMPACT_BANDS - 1 >= int(bands / 4):
        return 2
    elif COMPACT_BANDS - 1 >= int(bands / 8):
        return 3
    elif COMPACT_BANDS - 1 >= int(bands / 16):
        return 4
    elif COMPACT_BANDS - 1 >= int(bands / 32):
        return 5
    elif COMPACT_BANDS - 1 >= int(bands / 64):
        return 6
    elif COMPACT_BANDS - 1 >= int(bands / 128):
        return 7
    elif COMPACT_BANDS - 1 >= int(bands / 256):
        return 8

    raise ValueError(
        f"{bands} bands cannot produce {COMPACT_BANDS} compact bands."
    )


def resize_lr_to_full(lr_chw: torch.Tensor) -> torch.Tensor:
    # torchvision Resize((256,256)) on a tensor defaults to bilinear.
    return F.interpolate(
        lr_chw.unsqueeze(0),
        size=(OUT_SIZE, OUT_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)



def preprocess_gt(gt_path: str, mask_path: str | None, edge_path: str | None):

    gt_raw = tifffile.imread(gt_path)

    if gt_raw.ndim != 3:
        raise ValueError(
            f"Expected 3D HSI cube, got {gt_raw.shape}"
        )

    bands, raw_h, raw_w = gt_raw.shape

    if bands < COMPACT_BANDS:
        raise ValueError(
            f"Need >= {COMPACT_BANDS} spectral bands, got {bands}"
        )

    # --------------------------------------------------------
    # EXACT REPOSITORY ORDER
    #
    # 1. Create LR from the ORIGINAL spatial dimensions.
    # 2. Normalize.
    # 3. Resize HR and LR to 256x256.
    # --------------------------------------------------------

    gt_float = gt_raw.astype(np.float32)

    lr_raw = zoom(
        gt_float,
        (1.0, 0.25, 0.25),
        order=3,
    )

    gt_norm = gt_float / 10000.0
    lr_norm = lr_raw / 10000.0

    # --------------------------------------------------------
    # Convert CHW -> HWC and resize spatial dimensions
    # to the repository's output size.
    # --------------------------------------------------------

    hr_tensor = torch.from_numpy(
        gt_norm
    ).float()

    lr_tensor = torch.from_numpy(
        lr_norm
    ).float()

    # HR: arbitrary raw HxW -> 256x256
    hr_full = F.interpolate(
        hr_tensor.unsqueeze(0),
        size=(OUT_SIZE, OUT_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # LR: quarter-resolution -> 256x256
    lr_full = F.interpolate(
        lr_tensor.unsqueeze(0),
        size=(OUT_SIZE, OUT_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # --------------------------------------------------------
    # Flatten exactly like Dataset:
    #
    # preprocess1(image.transpose(1,2,0))
    #     -> [256,256,bands]
    #     -> reshape(bands, 256*256)
    #     -> transpose(0,1)
    # --------------------------------------------------------

    hr_flat = (
        hr_full
        .permute(1, 2, 0)
        .reshape(-1, bands)
    )

    lr_flat = (
        lr_full
        .permute(1, 2, 0)
        .reshape(-1, bands)
    )

    # --------------------------------------------------------
    # Same RWA level logic as repository Dataset
    # --------------------------------------------------------

    level = wavelet_level(bands)

    rwa_hr, w_hr = rwa(
        hr_flat,
        level,
        1,
    )

    rwa_lr, w_lr = rwa(
        lr_flat,
        level,
        1,
    )

    rwa_hr = np.asarray(
        rwa_hr,
        dtype=np.float32,
    )

    rwa_lr = np.asarray(
        rwa_lr,
        dtype=np.float32,
    )

    if rwa_hr.shape[1] < COMPACT_BANDS:
        raise ValueError(
            f"RWA HR produced only {rwa_hr.shape[1]} "
            f"bands; need {COMPACT_BANDS}"
        )

    if rwa_lr.shape[1] < COMPACT_BANDS:
        raise ValueError(
            f"RWA LR produced only {rwa_lr.shape[1]} "
            f"bands; need {COMPACT_BANDS}"
        )

    # --------------------------------------------------------
    # Compact representation
    # --------------------------------------------------------

    hr_hf = rwa_hr[
        :,
        :COMPACT_BANDS
    ]

    lr_hf = rwa_lr[
        :,
        :COMPACT_BANDS
    ]

    # --------------------------------------------------------
    # Per-sample PCA exactly matching existing Dataset
    # --------------------------------------------------------

    pca = PCA(
        n_components=COMPACT_BANDS
    )

    lr_pca = pca.fit_transform(
        lr_hf
    )

    hr_pca = pca.transform(
        hr_hf
    )

    # --------------------------------------------------------
    # Keep first 20 PCA channels
    # --------------------------------------------------------

    lr20 = lr_pca[
        :,
        :PCA_BANDS
    ].reshape(
        OUT_SIZE,
        OUT_SIZE,
        PCA_BANDS,
    )

    hr20 = hr_pca[
        :,
        :PCA_BANDS
    ].reshape(
        OUT_SIZE,
        OUT_SIZE,
        PCA_BANDS,
    )

    img_lr_hf = torch.from_numpy(
        lr20.transpose(2, 0, 1).copy()
    ).float() / 14000.0

    img_hr_hf = torch.from_numpy(
        hr20.transpose(2, 0, 1).copy()
    ).float() / 14000.0

    # --------------------------------------------------------
    # Mask / edge
    # --------------------------------------------------------

    if mask_path is not None and Path(mask_path).exists():

        mask_np = np.load(mask_path).astype(
            np.float32
        )

        mask = torch.from_numpy(
            mask_np
        )

        if tuple(mask.shape) != (
            OUT_SIZE,
            OUT_SIZE,
        ):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(OUT_SIZE, OUT_SIZE),
                mode="nearest",
            ).squeeze(0).squeeze(0)

    else:

        mask = torch.ones(
            OUT_SIZE,
            OUT_SIZE,
            dtype=torch.float32,
        )

    if edge_path is not None and Path(edge_path).exists():

        edge_np = np.load(edge_path).astype(
            np.float32
        )

        edge = torch.from_numpy(
            edge_np
        )

        if tuple(edge.shape) != (
            OUT_SIZE,
            OUT_SIZE,
        ):
            edge = F.interpolate(
                edge.unsqueeze(0).unsqueeze(0),
                size=(OUT_SIZE, OUT_SIZE),
                mode="nearest",
            ).squeeze(0).squeeze(0)

    else:

        edge = torch.ones(
            OUT_SIZE,
            OUT_SIZE,
            dtype=torch.float32,
        )

    # --------------------------------------------------------
    # Validate final representation
    # --------------------------------------------------------

    if img_lr_hf.shape != (
        PCA_BANDS,
        OUT_SIZE,
        OUT_SIZE,
    ):
        raise RuntimeError(
            f"Invalid LR latent shape: "
            f"{img_lr_hf.shape}"
        )

    if img_hr_hf.shape != (
        PCA_BANDS,
        OUT_SIZE,
        OUT_SIZE,
    ):
        raise RuntimeError(
            f"Invalid HR latent shape: "
            f"{img_hr_hf.shape}"
        )

    result = {
        "img_lr_hf": img_lr_hf.contiguous(),
        "img_hr_hf": img_hr_hf.contiguous(),
        "mask": mask.contiguous(),
        "edge": edge.contiguous(),
        "bands": bands,
        "raw_height": raw_h,
        "raw_width": raw_w,
        "rwa_level": level,
        "w": w_lr,
        "img_lr_recov": torch.from_numpy(
            lr_pca[
                :,
                PCA_BANDS:COMPACT_BANDS
            ].reshape(
                OUT_SIZE,
                OUT_SIZE,
                COMPACT_BANDS - PCA_BANDS,
            )
        ).float(),
    }

    # Release large temporary objects.
    del (
        gt_raw,
        gt_float,
        lr_raw,
        gt_norm,
        lr_norm,
        hr_tensor,
        lr_tensor,
        hr_full,
        lr_full,
        hr_flat,
        lr_flat,
        rwa_hr,
        rwa_lr,
        hr_hf,
        lr_hf,
        lr_pca,
        hr_pca,
        lr20,
        hr20,
    )

    gc.collect()

    return result


def fetch_and_preprocess(record, temp_root):
    temp_root = Path(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    gt_file = temp_root / Path(record["gt"]).name
    download_file(record["gt"], gt_file)

    mask_file = None
    edge_file = None

    try:
        if record.get("mask"):
            mask_file = temp_root / Path(record["mask"]).name
            download_file(record["mask"], mask_file)

        if record.get("edge"):
            edge_file = temp_root / Path(record["edge"]).name
            download_file(record["edge"], edge_file)

        sample = preprocess_gt(
            str(gt_file),
            str(mask_file) if mask_file else None,
            str(edge_file) if edge_file else None,
        )

        return sample

    finally:
        # No persistent HF cache accumulation.
        for p in [gt_file, mask_file, edge_file]:
            if p is not None and p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

        gc.collect()
