
import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from .model.edm import (
    ElucidatedDiffusion,
    UNet3DWithSpectralFidelity,
)

from .kaggle_hf_stream import (
    HF_REPO,
    OUT_SIZE,
    PCA_BANDS,
    load_manifest,
    fetch_and_preprocess,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "epoch_200.pth"
MANIFEST = ROOT / "hf_manifest" / "train_manifest.json"
OUTPUT_DIR = ROOT / "results" / "cfg_finetune_fp32"
SAMPLE_ORDER_FILE = OUTPUT_DIR / "sample_order.npy"
DATASET_STATE_FILE = OUTPUT_DIR / "dataset_state.json"
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
SAMPLES_PER_RUN = int(os.environ.get("GEW_SAMPLES_PER_RUN", "2000"))

if SAMPLES_PER_RUN <= 0:
    raise ValueError("GEW_SAMPLES_PER_RUN must be greater than 0")
RESUME = os.environ.get("GEW_RESUME", "")
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
PREFETCH_SIZE = 15


def prefetch_sample(record, temp_dir):
    fetch_start = time.perf_counter()

    try:
        sample = make_sample(
            record,
            temp_dir,
        )

        fetch_time = time.perf_counter() - fetch_start

        return {
            "ok": True,
            "sample": sample,
            "fetch_time": fetch_time,
            "error": None,
        }

    except Exception as exc:
        fetch_time = time.perf_counter() - fetch_start

        return {
            "ok": False,
            "sample": None,
            "fetch_time": fetch_time,
            "error": repr(exc),
        }


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
    diffusion,
    optimizer,
    scaler,
    epoch,
    step,
    last_loss,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        # Original GEWDiff checkpoint fields
        "unet_state_dict": model.state_dict(),
        "gaussian_diff_config": diffusion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": float(last_loss),

        # CFG-specific fields
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
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

    # Standard AdamW:
    # full-precision optimizer states for checkpoint compatibility.
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=LR,
        weight_decay=0.001,
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

    print("\nRunning scaled FP16 backward...")

    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=128.0,
    )

    scaler.scale(total_loss).backward()

    scaler.unscale_(optimizer)

    bad_grads = []

    for name, param in diffusion.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                bad_grads.append(name)

    if bad_grads:
        print("\nNON-FINITE GRADIENTS:")
        for name in bad_grads[:30]:
            print("  ", name)
        print("Total bad parameters:", len(bad_grads))
        raise RuntimeError(
            "GradScaler backward produced non-finite gradients"
        )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        diffusion.parameters(),
        1.0,
    )

    print("Scaled FP16 gradient norm:", float(grad_norm))

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

    if RESUME:
        resume_path = Path(RESUME)

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )

        missing, unexpected = model.load_state_dict(
            checkpoint["unet_state_dict"],
            strict=False,
        )

        if missing or unexpected:
            raise RuntimeError(
                "Resume checkpoint mismatch.\n"
                f"Missing: {len(missing)}\n"
                f"Unexpected: {len(unexpected)}"
            )

        print(
            f"Resuming from checkpoint: {resume_path}"
        )
    else:
        checkpoint = load_base_checkpoint(model)

    model = model.to(device)

    diffusion = build_diffusion(model).to(device)

    # Standard AdamW with full-precision optimizer states.
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=LR,
        weight_decay=0.001,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=128.0,
    )

    if world_size > 1:
        diffusion = DDP(
            diffusion,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    print_rank = rank == 0

    if RESUME:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        step = int(checkpoint["step"])
        start_epoch = int(checkpoint["epoch"])

        if print_rank:
            print(
                f"Resumed at step={step}, "
                f"start_epoch={start_epoch}"
            )
    else:
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

    # ------------------------------------------------------------------
    # Controlled sample/epoch progression
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if world_size != 1:
        raise RuntimeError(
            "GEW_SAMPLES_PER_RUN mode currently requires world_size=1."
        )

    # Create/load one deterministic shuffled order for the current epoch.
    if SAMPLE_ORDER_FILE.exists():
        order = np.load(SAMPLE_ORDER_FILE)

        if len(order) != len(manifest):
            raise RuntimeError(
                "Saved sample order does not match current manifest size."
            )
    else:
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(manifest)).astype(np.int64)
        np.save(SAMPLE_ORDER_FILE, order)

    # Load persistent dataset position.
    if DATASET_STATE_FILE.exists():

        with open(DATASET_STATE_FILE, "r") as f:
            dataset_state = json.load(f)

        current_epoch = int(dataset_state["epoch"])
        sample_cursor = int(dataset_state["sample_cursor"])

    else:

        current_epoch = 0
        sample_cursor = 0

    # If the previous epoch was completed, begin a new epoch.
    if sample_cursor >= len(manifest):

        current_epoch += 1
        sample_cursor = 0

        rng = np.random.default_rng(SEED + current_epoch)
        order = rng.permutation(len(manifest)).astype(np.int64)

        np.save(SAMPLE_ORDER_FILE, order)

    # Number of samples requested for THIS execution.
    run_start = sample_cursor
    run_end = min(
        run_start + SAMPLES_PER_RUN,
        len(manifest),
    )

    samples_this_run = run_end - run_start

    print(
        f"[rank {rank}] "
        f"Epoch {current_epoch + 1} | "
        f"samples {run_start}-{run_end - 1} | "
        f"count={samples_this_run} | "
        f"dataset={len(manifest)}"
    )

    # Save the intended run boundary.
    dataset_state = {
        "epoch": current_epoch,
        "sample_cursor": run_start,
        "run_end": run_end,
    }

    with open(DATASET_STATE_FILE, "w") as f:
        json.dump(dataset_state, f, indent=2)


    # ================================================================
    # TRAINING LOOP
    # ================================================================

    while sample_cursor < run_end:

        local_indices = list(
            order[run_start:run_end]
        )


        with ThreadPoolExecutor(
            max_workers=PREFETCH_SIZE
        ) as executor:

            futures = []

            # --------------------------------------------------
            # Initial prefetch
            # --------------------------------------------------

            initial_count = min(
                PREFETCH_SIZE,
                len(local_indices),
            )

            for i in range(initial_count):

                record = manifest[
                    int(local_indices[i])
                ]

                futures.append(
                    executor.submit(
                        prefetch_sample,
                        record,
                        TMP_DIR
                        / f"rank_{rank}"
                        / f"prefetch_{i}",
                    )
                )

            # --------------------------------------------------
            # Training + rolling prefetch
            # --------------------------------------------------

            for i, idx in enumerate(local_indices):

                if sample_cursor >= run_end:
                    break

                record = manifest[int(idx)]

                wait_start = time.perf_counter()

                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                future = done.pop()
                futures.remove(future)
                result = future.result()
                wait_time = (
                    time.perf_counter()
                    - wait_start
                )

                # --------------------------------------------------
                # Immediately submit the next sample
                # --------------------------------------------------

                next_i = i + PREFETCH_SIZE

                if next_i < len(local_indices):

                    next_record = manifest[
                        int(local_indices[next_i])
                    ]

                    futures.append(
                        executor.submit(
                            prefetch_sample,
                            next_record,
                            TMP_DIR
                            / f"rank_{rank}"
                            / f"prefetch_{next_i}",
                        )
                    )

                # --------------------------------------------------
                # Handle failed dataset sample
                # --------------------------------------------------

                if not result["ok"]:

                    if print_rank:
                        print(
                            f"[SKIP] sample={i} "
                            f"path={record['gt']} "
                            f"fetch_time={result['fetch_time']:.2f}s "
                            f"wait_time={wait_time:.2f}s "
                            f"error={result['error']}"
                        )

                    continue

                sample = result["sample"]
                fetch_time = result["fetch_time"]

                # --------------------------------------------------
                # Prepare batch
                # --------------------------------------------------

                lr, hr, mask, edge = prepare_batch(
                    sample,
                    device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # --------------------------------------------------
                # Forward + loss
                # --------------------------------------------------

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

                # --------------------------------------------------
                # Backward + optimizer
                # --------------------------------------------------

                scaler.scale(
                    total_loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    diffusion.parameters(),
                    1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                step += 1
                sample_cursor += 1

                # --------------------------------------------------
                # Print EVERY training step
                # --------------------------------------------------

                if print_rank:

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    steps_per_sec = (
                        step
                        / max(elapsed, 1e-6)
                    )

                    print(
                        f"step={step} "
                        f"loss={float(total_loss.detach()):.6f} "
                        f"pixel={float(loss1.detach()):.6f} "
                        f"perc={float(loss2.detach()):.6f} "
                        f"grad={float(loss3.detach()):.6f} "
                        f"fetch={fetch_time:.2f}s "
                        f"wait={wait_time:.2f}s "
                        f"steps/s={steps_per_sec:.4f}"
                    )

                # --------------------------------------------------
                # Save checkpoint
                # --------------------------------------------------

                if print_rank and (
                    step % SAVE_EVERY == 0
                    or sample_cursor == run_end
                ):

                    if hasattr(diffusion, "module"):
                        raw_model = diffusion.module.net
                    else:
                        raw_model = diffusion.net

                    path = save_checkpoint(
                        raw_model,
                        diffusion.module if isinstance(diffusion, DDP) else diffusion,
                        optimizer,
                        scaler,
                        current_epoch,
                        step,
                        float(total_loss.detach()),
                        OUTPUT_DIR,
                    )

                    print(
                        "Checkpoint:",
                        path,
                    )

                # --------------------------------------------------
                # Cleanup
                # --------------------------------------------------

                del sample
                del lr, hr, mask, edge
                if step % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Update dataset cursor
        # ------------------------------------------------------------------

        if print_rank:
            if sample_cursor >= len(manifest):

                # Entire dataset completed -> next run starts a new epoch.
                current_epoch += 1
                sample_cursor = 0

                rng = np.random.default_rng(
                    SEED + current_epoch
                )

                order = rng.permutation(
                    len(manifest)
                ).astype(np.int64)

                np.save(
                    SAMPLE_ORDER_FILE,
                    order
                )

                print(
                    f"[rank {rank}] "
                    f"Epoch {current_epoch} completed. "
                    f"Starting next epoch with a new shuffle."
                )

            else:

                print(
                    f"[rank {rank}] "
                    f"Run completed. "
                    f"Next sample cursor: {sample_cursor}"
                )

            dataset_state = {
                "epoch": current_epoch,
                "sample_cursor": sample_cursor,
                "run_end": sample_cursor,
            }

            with open(DATASET_STATE_FILE, "w") as f:
                json.dump(dataset_state, f, indent=2)
                
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
