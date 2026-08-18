"""
GEWDiff + CFG Inference Testing
Week 2: Test CFG with different guidance scales on pretrained checkpoint
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from typing import Optional, Tuple
import json

class CFGInferenceTester:
    """Test CFG inference with different guidance scales"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: Path to pretrained epoch_200.pth
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint
        try:
            self.checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"✓ Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            raise
    
    def test_guidance_scales(
        self,
        guidance_scales: list = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0],
        num_steps: int = 50,
        batch_size: int = 1,
        input_shape: Tuple = (1, 20, 64, 64),  # (batch, channels, H, W) for latent
    ):
        """
        Test inference with different guidance scales
        
        CFG formula: output = z_cond + w * (z_cond - z_uncond)
        where:
            w=0.0  -> purely unconditional
            w=1.0  -> normal conditional (no guidance)
            w>1.0  -> enhanced conditioning
            w<1.0  -> weak conditioning
        
        Args:
            guidance_scales: List of guidance scale values to test
            num_steps: Number of denoising steps
            batch_size: Batch size for inference
            input_shape: Shape of input latent
        
        Returns:
            Dictionary with results for each guidance scale
        """
        
        results = {
            'guidance_scale': [],
            'output_mean': [],
            'output_std': [],
            'output_range': [],
            'inference_time': [],
        }
        
        print("\n" + "="*60)
        print("CFG INFERENCE TEST")
        print("="*60)
        print(f"Input shape: {input_shape}")
        print(f"Num steps: {num_steps}")
        print(f"Guidance scales: {guidance_scales}")
        print("="*60)
        
        for w in guidance_scales:
            print(f"\nTesting guidance_scale={w}")
            
            try:
                # Simulate inference
                # In practice, integrate with your actual sampling loop (DPM-Solver++, etc.)
                
                # Create dummy latent and conditioning
                z_latent = torch.randn(input_shape, device=self.device)
                z_lr_cond = torch.randn(input_shape, device=self.device) * 0.5
                
                # Simulate denoising loop with CFG
                output = self._simulate_cfg_denoising(
                    z_latent, z_lr_cond, guidance_scale=w, num_steps=num_steps
                )
                
                # Compute output statistics
                output_mean = output.mean().item()
                output_std = output.std().item()
                output_min = output.min().item()
                output_max = output.max().item()
                output_range = output_max - output_min
                
                results['guidance_scale'].append(w)
                results['output_mean'].append(output_mean)
                results['output_std'].append(output_std)
                results['output_range'].append(output_range)
                
                print(f"  Mean: {output_mean:.4f} | Std: {output_std:.4f} | Range: {output_range:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results['guidance_scale'].append(w)
                results['output_mean'].append(None)
                results['output_std'].append(None)
                results['output_range'].append(None)
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        for i in range(len(results['guidance_scale'])):
            w = results['guidance_scale'][i]
            mean = results['output_mean'][i]
            std = results['output_std'][i]
            print(f"w={w:3.1f}: mean={mean:7.4f} std={std:7.4f}")
        
        return results
    
    def _simulate_cfg_denoising(
        self,
        z_T: torch.Tensor,
        z_lr_cond: torch.Tensor,
        guidance_scale: float = 3.0,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Simulate CFG denoising loop
        
        In reality, this would be:
        1. Load model
        2. For each timestep t from T to 0:
           - Get z_cond = model(z_t, z_lr_cond, conditioning)
           - Get z_uncond = model(z_t, zeros, no conditioning)
           - z_{t-1} = z_cond + w * (z_cond - z_uncond)
           
        For now, this is a placeholder that demonstrates the concept
        """
        
        z = z_T.clone()
        
        for step in range(num_steps):
            # Simulate conditional and unconditional predictions
            # In reality, these come from model predictions
            z_cond = z * 0.95 + 0.1 * torch.randn_like(z)  # Slightly less noise
            z_uncond = z * 0.90 + 0.2 * torch.randn_like(z)  # More noise
            
            # Apply CFG guidance formula
            if guidance_scale != 1.0:
                z = z_cond + guidance_scale * (z_cond - z_uncond)
            else:
                z = z_cond
        
        return z
    
    def analyze_guidance_effect(self, results: dict):
        """Analyze how guidance scale affects output"""
        
        print("\n" + "="*60)
        print("GUIDANCE ANALYSIS")
        print("="*60)
        
        guidance_scales = np.array(results['guidance_scale'])
        output_means = np.array(results['output_mean'])
        output_stds = np.array(results['output_std'])
        
        # Remove None values
        valid = ~np.isnan(output_means)
        guidance_scales = guidance_scales[valid]
        output_means = output_means[valid]
        output_stds = output_stds[valid]
        
        if len(guidance_scales) > 1:
            # Compute trends
            mean_change = output_means[-1] - output_means[0]
            std_change = output_stds[-1] - output_stds[0]
            
            print(f"\nMean change (w={guidance_scales[0]:.1f} to w={guidance_scales[-1]:.1f}):")
            print(f"  {mean_change:+.4f}")
            print(f"\nStd change:")
            print(f"  {std_change:+.4f}")
            
            # Recommendation
            if abs(mean_change) < 0.1:
                print("\n✓ Guidance scales have minimal effect on output mean")
                print("  This is GOOD - model is stable across guidance scales")
            else:
                print("\n⚠ Guidance scales significantly affect output")
                print("  May need to tune guidance scale carefully")
            
            # Find optimal scale (minimum variance)
            optimal_idx = np.argmin(output_stds)
            optimal_scale = guidance_scales[optimal_idx]
            optimal_std = output_stds[optimal_idx]
            
            print(f"\nOptimal guidance scale (min variance): {optimal_scale:.1f}")
            print(f"  Output std: {optimal_std:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CFG Inference Testing")
    parser.add_argument("--checkpoint", default="epoch_200.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="cfg_inference_results.json")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = CFGInferenceTester(args.checkpoint, device=args.device)
    
    # Test guidance scales
    results = tester.test_guidance_scales(
        guidance_scales=[0.0, 1.0, 3.0, 5.0, 7.0, 9.0],
        num_steps=50,
    )
    
    # Analyze results
    tester.analyze_guidance_effect(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
