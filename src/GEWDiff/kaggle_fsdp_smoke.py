
import os
import sys
import gc
from pathlib import Path

import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)

ROOT = Path("/kaggle/working/GEWDiff-BTP")
SRC = ROOT / "src" / "GEWDiff"

sys.path.insert(0, str(SRC))

from model.edm import (
    ElucidatedDiffusion,
    UNet3DWithSpectralFidelity,
)
from kaggle_hf_stream import (
    load_manifest,
    fetch_and_preprocess,
)

CHECKPOINT = ROOT / "checkpoints" / "epoch_200.pth"
MANIFEST = ROOT / "hf_manifest" / "train_manifest.json"
TMP = ROOT / "hf_tmp_fsdp"


def log(rank, msg):
    print(
        f"[rank {rank}] {msg}",
        flush=True,
    )


def setup():
    log(0, "Initializing distributed process group...")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device(
        "cuda",
        local_rank,
    )

    log(
        rank,
        f"Distributed initialized | "
        f"local_rank={local_rank} | "
        f"world={dist.get_world_size()}",
    )

    return rank, local_rank, device


def build_model():
    return UNet3DWithSpectralFidelity(
        sample_size=256,
        in_channels=41,
        out_channels=20,
        norm_type="group",
        layers_per_block=4,
        block_out_channels=(
            128,
            128,
            256,
            256,
            512,
            512,
        ),
        down_block_types=(
            "DownBlock3D",
            "DownBlock3D",
            "DownBlock3D",
            "DownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types=(
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "UpBlock3D",
            "UpBlock3D",
            "UpBlock3D",
            "UpBlock3D",
        ),
    )


def main():

    rank, local_rank, device = setup()

    log(
        rank,
        f"GPU = {torch.cuda.get_device_name(local_rank)}",
    )

    # --------------------------------------------------------
    # MODEL CONSTRUCTION
    # --------------------------------------------------------

    log(rank, "Building model on CPU...")
    model = build_model()

    log(
        rank,
        "Model built | "
        f"parameters={sum(p.numel() for p in model.parameters())}",
    )

    # --------------------------------------------------------
    # CHECKPOINT LOAD
    # --------------------------------------------------------

    if rank == 0:
        print("=" * 90, flush=True)
        print("FSDP VERBOSE REAL-HF SMOKE TEST", flush=True)
        print("=" * 90, flush=True)

    log(rank, "Loading checkpoint from CPU...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    log(
        rank,
        f"Checkpoint loaded | "
        f"epoch={checkpoint.get('epoch')} | "
        f"loss={checkpoint.get('loss')}",
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["unet_state_dict"],
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch | "
            f"missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )

    log(rank, "Checkpoint weights loaded.")

    del checkpoint
    gc.collect()

    # --------------------------------------------------------
    # FSDP WRAPPING
    # --------------------------------------------------------

    log(rank, "Creating FSDP wrapper...")

    mp = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=device,
        use_orig_params=True,
    )

    log(rank, "FSDP wrapper created.")

    # --------------------------------------------------------
    # DIFFUSION
    # --------------------------------------------------------

    log(rank, "Creating diffusion wrapper...")

    diffusion = ElucidatedDiffusion(
        model,
        image_size=256,
        channels=20,
        num_sample_steps=50,
        l1_lambda=0.8,
        l2_lambda=0.1,
        l3_lambda=0.1,
    ).to(device)

    diffusion.train()

    log(rank, "Diffusion wrapper ready.")

    dist.barrier()

    # --------------------------------------------------------
    # HF MANIFEST
    # --------------------------------------------------------

    log(rank, "Loading HF manifest...")

    manifest = load_manifest(MANIFEST)

    log(
        rank,
        f"Manifest loaded | {len(manifest)} samples",
    )

    record = manifest[0]

    log(
        rank,
        f"Downloading real HF sample: {record['gt']}",
    )

    sample = fetch_and_preprocess(
        record,
        TMP / f"rank_{rank}",
    )

    log(
        rank,
        "HF preprocessing complete | "
        f"LR={tuple(sample['img_lr_hf'].shape)} "
        f"HR={tuple(sample['img_hr_hf'].shape)}",
    )

    lr = sample["img_lr_hf"].unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    hr = sample["img_hr_hf"].unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    mask = sample["mask"].unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    edge = sample["edge"].unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    assert lr.shape == (1, 20, 256, 256)
    assert hr.shape == (1, 20, 256, 256)

    log(rank, "Real sample moved to GPU.")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=1e-5,
        weight_decay=0.001,
    )

    optimizer.zero_grad(set_to_none=True)

    dist.barrier()

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    log(rank, "Starting forward...")

    torch.cuda.reset_peak_memory_stats(device)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        loss, loss1, loss2, loss3 = diffusion(
            lr,
            hr,
            mask,
            edge,
        )

    log(
        rank,
        f"FORWARD PASS | loss={float(loss.detach())}",
    )

    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    log(rank, "Starting backward...")

    loss.backward()

    log(rank, "BACKWARD PASS")

    torch.nn.utils.clip_grad_norm_(
        diffusion.parameters(),
        1.0,
    )

    log(rank, "Gradient clipping complete.")

    optimizer.step()

    log(rank, "OPTIMIZER STEP COMPLETE")

    dist.barrier()

    peak = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 3)
    )

    log(
        rank,
        f"SUCCESS | peak VRAM={peak:.2f} GB",
    )

    dist.barrier()

    if rank == 0:
        print("=" * 90, flush=True)
        print(
            "FSDP 2-GPU REAL-HF STEP: SUCCESS",
            flush=True,
        )
        print("=" * 90, flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
