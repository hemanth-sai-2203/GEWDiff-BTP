
import os
import sys
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

ROOT = Path("/kaggle/working/GEWDiff-BTP")
SRC = ROOT / "src" / "GEWDiff"

sys.path.insert(0, str(SRC))

from model.edm import (
    ElucidatedDiffusion,
    UNet3DWithSpectralFidelity,
)

from diffusers.models.unets.unet_3d_blocks import (
    DownBlock3D,
    CrossAttnDownBlock3D,
    UpBlock3D,
    CrossAttnUpBlock3D,
    UNetMidBlock3DCrossAttn,
)

from kaggle_hf_stream import (
    load_manifest,
    fetch_and_preprocess,
)

CHECKPOINT = ROOT / "checkpoints" / "epoch_200.pth"
MANIFEST = ROOT / "hf_manifest" / "train_manifest.json"
OUTPUT = ROOT / "results" / "fsdp_cfg"
TMP = ROOT / "hf_tmp_fsdp_train"

MAX_STEPS = int(os.environ.get("GEW_MAX_STEPS", "1000"))
SAVE_EVERY = int(os.environ.get("GEW_SAVE_EVERY", "100"))
LR = float(os.environ.get("GEW_LR", "1e-5"))
P_DROP = 0.10


def log(rank, message):
    print(f"[rank {rank}] {message}", flush=True)


def setup():
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=local_rank,
    )

    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return rank, local_rank, device


class CFGElucidatedDiffusion(ElucidatedDiffusion):

    def __init__(self, *args, p_drop=P_DROP, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_drop = p_drop

    def forward(self, img_lr, images, mask=None, edge=None):

        if self.training and self.p_drop > 0:

            b = images.shape[0]

            drop = (
                torch.rand(
                    b,
                    device=images.device,
                )
                < self.p_drop
            )

            if drop.any():

                img_lr = img_lr.clone()
                img_lr[drop] = 0

                if mask is not None:
                    mask = mask.clone()
                    mask[drop] = 0

                if edge is not None:
                    edge = edge.clone()
                    edge[drop] = 0

        return super().forward(
            img_lr,
            images,
            mask,
            edge,
        )


def apply_safe_activation_checkpointing(model):
    """
    Checkpoint complete 3D UNet blocks.

    This does NOT rewrite UNet internals and therefore avoids
    the tensor-shape corruption caused by the previous manual
    forward patch.

    Parameters remain unchanged.
    """

    checkpoint_wrapper_fn = lambda module: checkpoint_wrapper(
        module,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_fn(module):
        return isinstance(
            module,
            (
                DownBlock3D,
                CrossAttnDownBlock3D,
                UpBlock3D,
                CrossAttnUpBlock3D,
                UNetMidBlock3DCrossAttn,
            ),
        )

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper_fn,
        check_fn=check_fn,
    )

    return model


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



def save_checkpoint(
    model,
    optimizer,
    step,
    last_loss,
):
    """
    Save a full CPU-offloaded model checkpoint from FSDP.

    This uses the PyTorch FSDP StateDict API compatible with
    the installed runtime instead of the nonexistent
    FSDP.full_state_dict() method.
    """

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Gather full model weights only on rank 0.
    # CPU offload prevents the gathered checkpoint from
    # unnecessarily occupying GPU memory.
    # --------------------------------------------------------

    state_cfg = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_cfg,
    ):
        full_state = model.state_dict()

    if dist.get_rank() == 0:

        payload = {
            "unet_state_dict": full_state,
            "epoch": 0,
            "step": int(step),
            "loss": float(last_loss),
            "cfg": {
                "p_drop": P_DROP,
                "lr": LR,
                "image_size": 256,
                "pca_bands": 20,
                "l1_lambda": 0.8,
                "l2_lambda": 0.1,
                "l3_lambda": 0.1,
            },
        }

        path = OUTPUT / f"cfg_step_{step:07d}.pth"

        torch.save(
            payload,
            path,
        )

        torch.save(
            payload,
            OUTPUT / "latest.pth",
        )

        print(
            f"Checkpoint saved: {path}",
            flush=True,
        )

    dist.barrier()


def main():

    rank, local_rank, device = setup()

    if rank == 0:
        print("=" * 90)
        print("GEWDIFF 2-GPU FSDP CFG FINETUNING")
        print("=" * 90)
        print("MAX_STEPS:", MAX_STEPS)
        print("LR:", LR)
        print("CFG p_drop:", P_DROP)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model()

    log(
        rank,
        f"Model created | params="
        f"{sum(p.numel() for p in model.parameters())}",
    )

    # --------------------------------------------------------
    # LOAD BASE CHECKPOINT
    # --------------------------------------------------------

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
            f"Checkpoint mismatch: "
            f"missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    log(
        rank,
        f"Base checkpoint loaded | "
        f"epoch={checkpoint.get('epoch')}",
    )

    del checkpoint
    gc.collect()

    # --------------------------------------------------------
    # FSDP
    # --------------------------------------------------------

    mixed_precision = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # --------------------------------------------------------
    # Activation checkpointing BEFORE FSDP
    # --------------------------------------------------------

    model = apply_safe_activation_checkpointing(
        model
    )

    log(
        rank,
        "Activation checkpointing applied to 3D UNet blocks"
    )

    # --------------------------------------------------------
    # FSDP
    # --------------------------------------------------------

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=device,
        use_orig_params=True,
    )

    log(rank, "FSDP FULL_SHARD ready")

    # --------------------------------------------------------
    # DIFFUSION
    # --------------------------------------------------------

    diffusion = CFGElucidatedDiffusion(
        model,
        image_size=256,
        channels=20,
        num_sample_steps=50,
        l1_lambda=0.8,
        l2_lambda=0.1,
        l3_lambda=0.1,
        p_drop=P_DROP,
    ).to(device)

    diffusion.train()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=LR,
        weight_decay=0.001,
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    manifest = load_manifest(MANIFEST)

    if not manifest:
        raise RuntimeError("HF manifest is empty.")

    log(
        rank,
        f"Training samples available: {len(manifest)}",
    )

    # Each rank receives a disjoint subset.
    world = dist.get_world_size()

    order = np.arange(len(manifest))

    step = 0
    epoch = 0

    start_time = time.time()

    while step < MAX_STEPS:

        # deterministic reshuffle per epoch
        rng = np.random.default_rng(
            42 + epoch
        )

        rng.shuffle(order)

        local_order = order[
            rank::world
        ]

        for sample_index in local_order:

            if step >= MAX_STEPS:
                break

            record = manifest[
                int(sample_index)
            ]

            try:

                # ------------------------------------------------
                # STREAM ONE SAMPLE
                # ------------------------------------------------

                sample = fetch_and_preprocess(
                    record,
                    TMP / f"rank_{rank}",
                )

                lr = sample[
                    "img_lr_hf"
                ].unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                hr = sample[
                    "img_hr_hf"
                ].unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                mask = sample[
                    "mask"
                ].unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                edge = sample[
                    "edge"
                ].unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                assert (
                    lr.shape
                    == (1, 20, 256, 256)
                )

                assert (
                    hr.shape
                    == (1, 20, 256, 256)
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # ------------------------------------------------
                # FORWARD
                # ------------------------------------------------

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

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at step {step}: "
                        f"{loss.item()}"
                    )

                # ------------------------------------------------
                # BACKWARD
                # ------------------------------------------------

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    diffusion.parameters(),
                    1.0,
                )

                optimizer.step()

                step += 1

                # ------------------------------------------------
                # LOG
                # ------------------------------------------------

                if rank == 0 and (
                    step == 1
                    or step % 5 == 0
                ):

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    rate = (
                        step
                        / max(
                            elapsed,
                            1e-6,
                        )
                    )

                    peak = (
                        torch.cuda.max_memory_allocated()
                        / (1024 ** 3)
                    )

                    print(
                        f"step={step}/{MAX_STEPS} "
                        f"loss={float(loss.detach()):.6f} "
                        f"pixel={float(loss1.detach()):.6f} "
                        f"perc={float(loss2.detach()):.6f} "
                        f"grad={float(loss3.detach()):.6f} "
                        f"steps/s={rate:.4f} "
                        f"VRAM={peak:.2f}GB "
                        f"sample={record['gt']}",
                        flush=True,
                    )

                # ------------------------------------------------
                # SAVE
                # ------------------------------------------------

                if (
                    step % SAVE_EVERY == 0
                    or step == MAX_STEPS
                ):

                    save_checkpoint(
                        model,
                        optimizer,
                        step,
                        loss.detach(),
                    )

                del sample
                del lr
                del hr
                del mask
                del edge

                gc.collect()

                torch.cuda.empty_cache()

            except Exception:

                log(
                    rank,
                    f"FAILED sample={record['gt']} "
                    f"at step={step}",
                )

                raise

        epoch += 1

    dist.barrier()

    if rank == 0:

        print("=" * 90)
        print("CFG FINETUNING COMPLETE")
        print("=" * 90)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
