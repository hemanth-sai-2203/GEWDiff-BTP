
import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import bitsandbytes as bnb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.edm import (
    ElucidatedDiffusion,
    UNet3DWithSpectralFidelity,
)

from kaggle_hf_stream import (
    HF_REPO,
    OUT_SIZE,
    PCA_BANDS,
    load_manifest,
    fetch_and_preprocess,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "epoch_200.pth"
MANIFEST = ROOT / "hf_manifest" / "train_manifest.json"
OUTPUT_DIR = ROOT / "results" / "cfg_finetune"
TMP_DIR = ROOT / "hf_tmp"


# Official released training values.
COMPACT_BANDS = 121
L1 = 0.8
L2 = 0.1
L3 = 0.1

# CFG modification.
P_DROP = 0.10

# Fine-tuning default is deliberately configurable.
LR = float(os.environ.get("GEW_FINETUNE_LR", "1e-5"))
MAX_STEPS = int(os.environ.get("GEW_MAX_STEPS", "1000"))
SAVE_EVERY = int(os.environ.get("GEW_SAVE_EVERY", "100"))
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CFGElucidatedDiffusion(ElucidatedDiffusion):

    def __init__(self, *args, p_drop=P_DROP, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_drop = float(p_drop)

    def forward(self, img_lr, images, mask=None, edge=None):
        batch = images.shape[0]

        if self.training and self.p_drop > 0:
            drop = torch.rand(
                batch,
                device=images.device,
            ) < self.p_drop

            if drop.any():
                img_lr = img_lr.clone()
                img_lr[drop] = 0

                if mask is not None:
                    mask = mask.clone()
                    mask[drop] = 0

        return super().forward(
            img_lr,
            images,
            mask,
            edge,
        )


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda:0")

    return world_size, rank, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_model():
    model = UNet3DWithSpectralFidelity(
        sample_size=256,
        in_channels=41,
        out_channels=20,
        norm_type="group",
        layers_per_block=4,
        block_out_channels=(
            128, 128, 256, 256, 512, 512
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

    return model


def build_diffusion(model):
    return CFGElucidatedDiffusion(
        model,
        image_size=OUT_SIZE,
        channels=PCA_BANDS,
        num_sample_steps=50,
        l1_lambda=L1,
        l2_lambda=L2,
        l3_lambda=L3,
        p_drop=P_DROP,
    )


def load_base_checkpoint(model):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["unet_state_dict"],
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint mismatch.\n"
            f"Missing: {len(missing)}\n"
            f"Unexpected: {len(unexpected)}"
        )

    return checkpoint


def make_sample(record, temp_dir):
    return fetch_and_preprocess(
        record,
        temp_dir,
    )


def prepare_batch(sample, device):
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

    if lr.shape != (1, 20, 256, 256):
        raise RuntimeError(f"Unexpected LR latent: {lr.shape}")

    if hr.shape != (1, 20, 256, 256):
        raise RuntimeError(f"Unexpected HR latent: {hr.shape}")

    if mask.shape != (1, 256, 256):
        raise RuntimeError(f"Unexpected mask: {mask.shape}")

    if edge.shape != (1, 256, 256):
        raise RuntimeError(f"Unexpected edge: {edge.shape}")

    return lr, hr, mask, edge


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    step,
    last_loss,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "unet_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": float(last_loss),
        "cfg": {
            "p_drop": P_DROP,
            "compact_bands": COMPACT_BANDS,
            "pca_bands": PCA_BANDS,
            "l1": L1,
            "l2": L2,
            "l3": L3,
            "lr": LR,
        },
    }

    path = output_dir / f"cfg_step_{step:07d}.pth"

    torch.save(
        state,
        path,
    )

    latest = output_dir / "latest.pth"

    torch.save(
        state,
        latest,
    )

    return path


@torch.no_grad()
def validate_conditioning(diffusion, lr, mask):
    # This directly checks the 41-channel EDM assembly.
    sigma = torch.ones(
        lr.shape[0],
        device=lr.device,
        dtype=torch.float32,
    )

    noised = torch.randn_like(lr)

    denoised = diffusion.preconditioned_network_forward(
        noised,
        lr,
        sigma,
        mask,
    )

    if denoised.shape != lr.shape:
        raise RuntimeError(
            f"Invalid denoised shape: {denoised.shape}"
        )

    return denoised


def smoke_test():
    print("=" * 90)
    print("GEWDIFF CFG SMOKE TEST")
    print("=" * 90)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required.")

    seed_everything(SEED)

    device = torch.device("cuda:0")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", CHECKPOINT)

    manifest = load_manifest(MANIFEST)

    if not manifest:
        raise RuntimeError("HF manifest is empty.")

    # Use a deterministic real sample.
    record = manifest[0]

    print("HF sample:", record["gt"])

    model = build_model()

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", parameter_count)

    checkpoint = load_base_checkpoint(model)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint loss:", checkpoint["loss"])
    print("STRICT MODEL CHECK: PASS")

    model = model.to(
        device=device,
        dtype=torch.float32,
    )

    diffusion = build_diffusion(model).to(device)

    diffusion.train()

    print("Downloading + preprocessing real HF sample...")

    sample = make_sample(
        record,
        TMP_DIR / "smoke",
    )

    print(
        "LR:",
        tuple(sample["img_lr_hf"].shape),
    )
    print(
        "HR:",
        tuple(sample["img_hr_hf"].shape),
    )
    print(
        "MASK:",
        tuple(sample["mask"].shape),
    )
    print(
        "EDGE:",
        tuple(sample["edge"].shape),
    )
    print("RAW BANDS:", sample["bands"])
    print("RWA LEVEL:", sample["rwa_level"])

    lr, hr, mask, edge = prepare_batch(
        sample,
        device,
    )

    print("\nChecking exact 41-channel construction...")

    _ = validate_conditioning(
        diffusion,
        lr,
        mask,
    )

    print("41-channel conditioning path: PASS")

    # T4-safe AdamW:
    # same AdamW optimization rule, 8-bit optimizer states.
    optimizer = bnb.optim.AdamW8bit(
        diffusion.parameters(),
        lr=LR,
        weight_decay=0.001,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    torch.cuda.reset_peak_memory_stats()

    print("\nRunning REAL forward + loss...")

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        total_loss, loss1, loss2, loss3 = diffusion(
            lr,
            hr,
            mask,
            edge,
        )

    if not torch.isfinite(total_loss):
        raise RuntimeError(
            f"Non-finite loss: {total_loss.item()}"
        )

    print("Total loss:", float(total_loss.detach()))
    print("Pixel loss:", float(loss1.detach()))
    print("Perceptual:", float(loss2.detach()))
    print("Gradient:", float(loss3.detach()))

    print("\nRunning backward...")

    scaler.scale(total_loss).backward()

    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        diffusion.parameters(),
        1.0,
    )

    if not torch.isfinite(grad_norm):
        raise RuntimeError(
            f"Non-finite gradient norm: {grad_norm}"
        )

    scaler.step(optimizer)
    scaler.update()

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print("Gradient norm:", float(grad_norm))
    print(f"Peak VRAM: {peak_vram:.2f} GB")

    print("\n" + "=" * 90)
    print("FULL REAL-HF CFG TRAINING STEP: PASS")
    print("=" * 90)

    del diffusion
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required.")

    seed_everything(SEED)

    world_size, rank, local_rank, device = setup_distributed()

    if rank == 0:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest = load_manifest(MANIFEST)

    if not manifest:
        raise RuntimeError("No training samples found.")

    model = build_model()

    checkpoint = load_base_checkpoint(model)

    model = model.to(device)

    diffusion = build_diffusion(model).to(device)

    # T4-safe AdamW:
    # same AdamW optimization rule, 8-bit optimizer states.
    optimizer = bnb.optim.AdamW8bit(
        diffusion.parameters(),
        lr=LR,
        weight_decay=0.001,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
    )

    if world_size > 1:
        diffusion = DDP(
            diffusion,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    print_rank = rank == 0

    step = 0
    start_epoch = 0

    total_per_epoch = (
        len(manifest) + world_size - 1
    ) // world_size

    print(
        f"[rank {rank}] samples={len(manifest)} "
        f"world={world_size}"
    )

    start_time = time.time()

    while step < MAX_STEPS:

        epoch = start_epoch

        rng = np.random.default_rng(
            SEED + epoch
        )

        order = rng.permutation(
            len(manifest)
        )

        local_order = order[
            rank::world_size
        ]

        for idx in local_order:

            if step >= MAX_STEPS:
                break

            record = manifest[int(idx)]

            try:
                sample = make_sample(
                    record,
                    TMP_DIR / f"rank_{rank}",
                )

                lr, hr, mask, edge = prepare_batch(
                    sample,
                    device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    total_loss, loss1, loss2, loss3 = diffusion(
                        lr,
                        hr,
                        mask,
                        edge,
                    )

                if not torch.isfinite(total_loss):
                    raise RuntimeError(
                        f"Non-finite loss at step {step}"
                    )

                scaler.scale(total_loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    diffusion.parameters(),
                    1.0,
                )

                scaler.step(optimizer)
                scaler.update()

                step += 1

                if print_rank and (
                    step % 10 == 0
                    or step == 1
                ):
                    elapsed = time.time() - start_time
                    steps_per_sec = step / max(elapsed, 1e-6)

                    print(
                        f"step={step}/{MAX_STEPS} "
                        f"loss={float(total_loss.detach()):.6f} "
                        f"pixel={float(loss1.detach()):.6f} "
                        f"perc={float(loss2.detach()):.6f} "
                        f"grad={float(loss3.detach()):.6f} "
                        f"steps/s={steps_per_sec:.4f}"
                    )

                if print_rank and (
                    step % SAVE_EVERY == 0
                    or step == MAX_STEPS
                ):
                    raw_model = (
                        diffusion.module.net
                        if isinstance(diffusion, DDP)
                        else diffusion.net
                    )

                    path = save_checkpoint(
                        raw_model,
                        optimizer,
                        scaler,
                        epoch,
                        step,
                        float(total_loss.detach()),
                        OUTPUT_DIR,
                    )

                    print("Checkpoint:", path)

                del sample
                del lr, hr, mask, edge
                gc.collect()

                # Release unused CUDA blocks between samples.
                torch.cuda.empty_cache()

            except Exception:
                print(
                    f"[rank {rank}] FAILURE on "
                    f"{record['gt']}"
                )
                raise

        start_epoch += 1

    cleanup_distributed()

    if print_rank:
        print("\nTRAINING COMPLETE")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["smoke", "train"],
        default="smoke",
    )

    args = parser.parse_args()

    if args.mode == "smoke":
        smoke_test()
    else:
        train()


if __name__ == "__main__":
    main()
