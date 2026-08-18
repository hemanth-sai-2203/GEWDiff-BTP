"""
CFG Fine-tuning Wrapper for GEWDiff
Classifier-Free Guidance: Train with 10% probability of dropping conditioning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class CFGElucidatedDiffusion(nn.Module):
    """
    Wrapper around existing ElucidatedDiffusion model to add CFG support
    
    CFG works by:
    1. Training: Randomly drop conditioning (z_LR, mask) with probability p_drop
    2. Inference: Blend conditional and unconditional outputs using guidance_scale w
       output = z_cond + w * (z_cond - z_uncond)
    """
    
    def __init__(self, base_diffusion, p_drop=0.1):
        """
        Args:
            base_diffusion: The original ElucidatedDiffusion model
            p_drop: Probability of dropping conditioning during training (default 0.1 = 10%)
        """
        super().__init__()
        self.base_diffusion = base_diffusion
        self.p_drop = p_drop
        self.training_mode = True
        
    def forward(self, x, x_r, mask=None, edge=None, drop_conditioning=None):
        """
        Forward pass with CFG support
        
        Args:
            x: LR image (conditioning signal z_LR)
            x_r: HR residual image
            mask: NDVI mask
            edge: Edge map
            drop_conditioning: If None, randomly decide based on p_drop during training
                              If True/False, explicitly drop/keep conditioning
        
        Returns:
            loss, loss1, loss2, loss3 (same as base_diffusion)
        """
        
        if self.training_mode and drop_conditioning is None:
            # During training, randomly drop conditioning with probability p_drop
            drop_conditioning = np.random.rand() < self.p_drop
        elif not self.training_mode:
            # During inference, don't drop (handled separately in inference)
            drop_conditioning = False
        
        if drop_conditioning:
            # Zero out conditioning signals
            x_cond = torch.zeros_like(x)
            mask_cond = torch.zeros_like(mask) if mask is not None else None
            edge_cond = torch.zeros_like(edge) if edge is not None else None
        else:
            # Keep conditioning signals
            x_cond = x
            mask_cond = mask
            edge_cond = edge
        
        # Call base diffusion with (possibly dropped) conditioning
        loss, loss1, loss2, loss3 = self.base_diffusion(
            x_cond, x_r, mask_cond, edge_cond
        )
        
        return loss, loss1, loss2, loss3
    
    def train(self, mode=True):
        """Set training mode"""
        super().train(mode)
        self.training_mode = mode
        self.base_diffusion.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def __getattr__(self, name):
        """Forward attribute access to base_diffusion"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_diffusion, name)


class CFGInferenceWrapper:
    """
    Inference wrapper for CFG with guidance scale blending
    
    Classifier-Free Guidance formula:
        output = z_cond + w * (z_cond - z_uncond)
    where:
        z_cond = output with conditioning
        z_uncond = output without conditioning
        w = guidance_scale (typically 3.0-7.5)
        w=0 -> only unconditional
        w=1 -> normal conditional
        w>1 -> exaggerates conditioning effect
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: The diffusion model (must support forward pass)
            device: GPU device
        """
        self.model = model
        self.device = device
    
    def sample_with_cfg(
        self,
        shape: Tuple[int, ...],
        conditioning: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        edge: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        num_steps: int = 50,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Sample from model using CFG guidance
        
        Args:
            shape: Output shape (batch_size, channels, height, width)
            conditioning: LR image (z_LR) - used as conditioning
            mask: NDVI mask
            edge: Edge map
            guidance_scale: CFG guidance strength (0=unconditional, 1=normal, >1=enhanced)
            num_steps: Number of denoising steps
            show_progress: Whether to show progress bar
        
        Returns:
            Sampled output with guidance
        """
        
        if guidance_scale == 1.0:
            # Standard sampling without CFG
            return self._sample_normal(shape, conditioning, mask, edge, num_steps, show_progress)
        
        # CFG sampling: blend conditional and unconditional
        z_cond = self._sample_normal(
            shape, conditioning, mask, edge, num_steps, show_progress
        )
        
        z_uncond = self._sample_normal(
            shape, None, None, None, num_steps, show_progress
        )
        
        # Apply CFG guidance formula
        output = z_cond + guidance_scale * (z_cond - z_uncond)
        
        return output
    
    def _sample_normal(
        self,
        shape: Tuple[int, ...],
        conditioning: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        edge: Optional[torch.Tensor],
        num_steps: int,
        show_progress: bool,
    ) -> torch.Tensor:
        """Internal helper: sample with or without conditioning"""
        
        # This is a placeholder - you'll need to integrate with your model's sampling loop
        # The key is that when conditioning is None, the model should output unconditional samples
        raise NotImplementedError(
            "Integrate with your model's actual sampling loop (DPM-Solver++ or similar)"
        )


# ============================================================================
# Usage Example:
# ============================================================================
#
# from model.edm import ElucidatedDiffusion
# from cfg_finetuning_wrapper import CFGElucidatedDiffusion
#
# # Load pretrained model
# base_model = ElucidatedDiffusion(...)
# base_model.load_state_dict(torch.load('epoch_200.pth'))
#
# # Wrap with CFG
# cfg_model = CFGElucidatedDiffusion(base_model, p_drop=0.1)
# cfg_model.train()
#
# # Training loop now automatically:
# # - Drops conditioning 10% of the time
# # - Calls base_diffusion forward with (possibly zero) conditioning
#
# for batch in train_loader:
#     loss, loss1, loss2, loss3 = cfg_model(
#         batch['img_lr_hf'],
#         batch['img_hr_hf'],
#         batch['mask'],
#         batch['edge']
#     )
#     loss.backward()
#     optimizer.step()
#
# # Inference with guidance scale w
# cfg_inference = CFGInferenceWrapper(cfg_model, device='cuda')
# output = cfg_inference.sample_with_cfg(
#     shape=(1, 20, 64, 64),
#     conditioning=lr_image,
#     guidance_scale=3.0,  # Values: 3.0-7.5 typical
#     num_steps=50
# )
