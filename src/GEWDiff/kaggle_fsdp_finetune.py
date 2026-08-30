
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
    FullOptimStateDictConfig,
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
CFG_RESUME = ROOT / "results" / "fsdp_cfg" / "latest.pth"
MANIFEST = ROOT / "hf_manifest" / "train_manifest.json"
OUTPUT = ROOT / "results" / "fsdp_cfg"
TMP = ROOT / "hf_tmp_fsdp_train"

MAX_STEPS = int(os.environ.get("GEW_MAX_STEPS", "4000"))
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
    diffusion,
    perceptual_optimizer,
    step,
    last_loss,
):
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_state_cfg = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    optim_state_cfg = FullOptimStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    # --------------------------------------------------------
    # FSDP UNET MODEL + OPTIMIZER
    # --------------------------------------------------------

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        model_state_cfg,
        optim_state_cfg,
    ):
        full_model_state = model.state_dict()

        full_optimizer_state = FSDP.optim_state_dict(
            model,
            optimizer,
        )

    # --------------------------------------------------------
    # NON-FSDP CONV_ADJUST
    # --------------------------------------------------------

    conv_adjust_state = {
        key: value.detach().cpu().clone()
        for key, value
        in diffusion.perceptual_loss.conv_adjust.state_dict().items()
    }

    perceptual_optimizer_state = (
        perceptual_optimizer.state_dict()
    )

    if dist.get_rank() == 0:

        payload = {
            "unet_state_dict": full_model_state,
            "unet_optimizer_state_dict": full_optimizer_state,

            "conv_adjust_state_dict": conv_adjust_state,
            "conv_adjust_optimizer_state_dict":
                perceptual_optimizer_state,

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

        path = OUTPUT / "latest.pth"

        tmp_path = OUTPUT / "latest.tmp.pth"

        # Write atomically so a failed save cannot leave a
        # corrupted latest.pth.
        torch.save(
            payload,
            tmp_path,
        )

        os.replace(
            tmp_path,
            path,
        )

        print(
            f"Checkpoint saved: {path} | step={step}",
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
    # LOAD RESUME / BASE CHECKPOINT
    # --------------------------------------------------------

    resume_step = 0

    if CFG_RESUME is not None and CFG_RESUME.exists():
        checkpoint = torch.load(
            CFG_RESUME,
            map_location="cpu",
            weights_only=False,
        )

        resume_step = int(checkpoint.get("step", 0))

        log(
            rank,
            f"CFG checkpoint loaded | step={resume_step}",
        )

    else:
        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        log(
            rank,
            f"Base checkpoint loaded | epoch={checkpoint.get('epoch')}",
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
    )

    # The FSDP-wrapped UNet is already on its CUDA device.
    # Move only the non-FSDP perceptual module.
    diffusion.perceptual_loss.to(device)

    diffusion.train()

    # --------------------------------------------------------
    # FREEZE PRETRAINED VGG FEATURE EXTRACTOR
    # --------------------------------------------------------

    for param in diffusion.perceptual_loss.feature_layers.parameters():
        param.requires_grad = False

    log(
        rank,
        "Pretrained VGG feature extractor frozen"
    )

    # --------------------------------------------------------
    # OPTIMIZERS
    # --------------------------------------------------------
    #
    # FSDP optimizer:
    #   Only parameters belonging to the FSDP-wrapped UNet.
    #
    # Perceptual optimizer:
    #   Only the trainable conv_adjust layer, which is outside FSDP.
    #
    # Keeping them separate is required because
    # FSDP.optim_state_dict() can only process parameters
    # belonging to the FSDP-wrapped model.
    # --------------------------------------------------------

# --------------------------------------------------------
# OPTIMIZERS
# --------------------------------------------------------

# FSDP-managed UNet parameters ONLY.
# The optimizer must not contain parameters belonging to
# modules outside the FSDP-wrapped UNet.
    unet_trainable_params = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        unet_trainable_params,
        lr=LR,
        weight_decay=0.001,
    )

    # conv_adjust is outside FSDP and therefore has its own
    # ordinary optimizer.
    conv_adjust_params = [
        param
        for param in diffusion.perceptual_loss.conv_adjust.parameters()
        if param.requires_grad
    ]

    perceptual_optimizer = torch.optim.AdamW(
        conv_adjust_params,
        lr=LR,
        weight_decay=0.001,
    )

    log(
        rank,
        f"UNet optimizer parameters: "
        f"{sum(p.numel() for p in unet_trainable_params):,}"
    )

    log(
        rank,
        f"Perceptual optimizer parameters: "
        f"{sum(p.numel() for p in conv_adjust_params):,}"
    )

    # --------------------------------------------------------
    # RESTORE OPTIMIZER STATE WHEN RESUMING CFG CHECKPOINT
    # --------------------------------------------------------

    if resume_step > 0:

        unet_optimizer_state = checkpoint.get(
            "unet_optimizer_state_dict"
        )

        conv_adjust_state = checkpoint.get(
            "conv_adjust_state_dict"
        )

        conv_adjust_optimizer_state = checkpoint.get(
            "conv_adjust_optimizer_state_dict"
        )

        if (
            unet_optimizer_state is None
            or conv_adjust_state is None
            or conv_adjust_optimizer_state is None
        ):
            raise RuntimeError(
                "CFG checkpoint is incomplete: "
                "expected UNet optimizer, conv_adjust, "
                "and conv_adjust optimizer states."
            )

        # --------------------------------------------------------
        # RESTORE FSDP UNET OPTIMIZER
        # --------------------------------------------------------

        unet_optimizer_state = (
            FSDP.optim_state_dict_to_load(
                model,
                optimizer,
                unet_optimizer_state,
            )
        )

        optimizer.load_state_dict(
            unet_optimizer_state
        )

        # --------------------------------------------------------
        # RESTORE CONV_ADJUST
        # --------------------------------------------------------

        diffusion.perceptual_loss.conv_adjust.load_state_dict(
            conv_adjust_state
        )

        # --------------------------------------------------------
        # RESTORE CONV_ADJUST OPTIMIZER
        # --------------------------------------------------------

        perceptual_optimizer.load_state_dict(
            conv_adjust_optimizer_state
        )

        log(
            rank,
            f"Both optimizer states + conv_adjust restored "
            f"| step={resume_step}",
        )



    # Checkpoint is no longer needed after model/optimizer restore.
    del checkpoint
    gc.collect()

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

    step = resume_step
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
                # DATA ERRORS ARE SAFELY SKIPPED
                # ACROSS ALL FSDP RANKS.
                # ------------------------------------------------

                sample = None
                fetch_ok = True

                try:
                    sample = fetch_and_preprocess(
                        record,
                        TMP / f"rank_{rank}",
                    )

                except Exception as exc:
                    fetch_ok = False

                    log(
                        rank,
                        f"SKIP bad sample={record['gt']} | "
                        f"{type(exc).__name__}: {exc}",
                    )

                # ------------------------------------------------
                # IMPORTANT:
                # Both FSDP ranks must make the SAME decision.
                #
                # If rank 1 gets a bad sample while rank 0 gets
                # a valid sample, rank 0 must NOT enter forward()
                # because FSDP collectives would otherwise mismatch.
                #
                # MIN -> if either rank failed, BOTH skip.
                # ------------------------------------------------

                fetch_flag = torch.tensor(
                    1 if fetch_ok else 0,
                    device=device,
                    dtype=torch.int32,
                )

                dist.all_reduce(
                    fetch_flag,
                    op=dist.ReduceOp.MIN,
                )

                if fetch_flag.item() == 0:

                    if sample is not None:
                        del sample

                    gc.collect()
                    torch.cuda.empty_cache()

                    continue

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

                optimizer.zero_grad(set_to_none=True)
                perceptual_optimizer.zero_grad(set_to_none=True)

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
                perceptual_optimizer.step()
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
                        diffusion,
                        perceptual_optimizer,
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
