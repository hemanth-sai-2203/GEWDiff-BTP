"""
GEWDiff Parameter Ablation Testing Script
Test rho, sigma_min, sigma_max on WDC dataset
Run on Kaggle T4 free tier
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm
import argparse
import json

# Add paths
sys.path.insert(0, '../input/gewdiff-btp/src/GEWDiff')
sys.path.insert(0, '../input/gewdiff-btp/src/GEWDiff/data')
sys.path.insert(0, '../input/gewdiff-btp/src/GEWDiff/model')
sys.path.insert(0, '../input/gewdiff-btp/src/GEWDiff/utils')

try:
    from dataset import Dataset
    from eval import quality_assessment, compare_mpsnr, compare_mssim, compare_sam, compare_ergas, compare_fid
    import tifffile
except Exception as e:
    print(f"Import error: {e}")
    print("Make sure dataset.py and eval.py are in path")


class AblationTester:
    """Test different parameter combinations"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: Path to epoch_200.pth checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.results = []
        
    def test_parameters(
        self,
        data_dir,
        rho_values=[0.5, 0.6, 0.7, 0.8, 0.9],
        sigma_min_values=[0.002, 0.02, 0.05, 0.1, 0.2],
        sigma_max_values=[60, 80, 100],
        num_steps=50,
    ):
        """
        Test all combinations of parameters
        
        Args:
            data_dir: Path to WDC test data
            rho_values: Noise schedule parameter rho
            sigma_min_values: Minimum sigma (noise level)
            sigma_max_values: Maximum sigma
            num_steps: Number of denoising steps
        
        Returns:
            DataFrame with results
        """
        
        total_tests = len(rho_values) * len(sigma_min_values) * len(sigma_max_values)
        pbar = tqdm(total=total_tests, desc="Ablation Study")
        
        for rho in rho_values:
            for sigma_min in sigma_min_values:
                for sigma_max in sigma_max_values:
                    
                    result = self._test_single_config(
                        data_dir,
                        rho=rho,
                        sigma_min=sigma_min,
                        sigma_max=sigma_max,
                        num_steps=num_steps,
                    )
                    
                    self.results.append(result)
                    pbar.update(1)
        
        pbar.close()
        
        return pd.DataFrame(self.results)
    
    def _test_single_config(
        self,
        data_dir,
        rho=0.7,
        sigma_min=0.002,
        sigma_max=80,
        num_steps=50,
    ):
        """Test single parameter configuration"""
        
        print(f"\nTesting: rho={rho}, sigma_min={sigma_min}, sigma_max={sigma_max}")
        
        try:
            # Load checkpoint with modified parameters
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Extract model state and config
            model_state = checkpoint.get('unet_state_dict') or checkpoint
            
            # TODO: Load your diffusion model here
            # model = ElucidatedDiffusion(rho=rho, sigma_min=sigma_min, sigma_max=sigma_max, ...)
            # model.load_state_dict(model_state)
            # model.to(self.device)
            # model.eval()
            
            # For now, this is a template - you'll integrate with your model
            metrics = {
                'rho': rho,
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'psnr': np.random.rand() * 10 + 20,  # Placeholder
                'ssim': np.random.rand() * 0.5 + 0.5,  # Placeholder
                'sam': np.random.rand() * 15,  # Placeholder (lower is better)
                'fid': np.random.rand() * 50,  # Placeholder
                'ergas': np.random.rand() * 10,  # Placeholder
                'inference_time': np.random.rand() * 30,  # Placeholder
            }
            
            print(f"  PSNR: {metrics['psnr']:.4f}, SAM: {metrics['sam']:.4f}, FID: {metrics['fid']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return {
                'rho': rho,
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'psnr': None,
                'ssim': None,
                'sam': None,
                'fid': None,
                'ergas': None,
                'inference_time': None,
                'error': str(e),
            }
    
    def save_results(self, output_path):
        """Save results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("ABLATION RESULTS SUMMARY")
        print("="*60)
        print(df.to_string())
        
        # Best results
        if 'sam' in df.columns:
            best_sam = df.loc[df['sam'].idxmin()]
            print(f"\nBest SAM: {best_sam['sam']:.4f}")
            print(f"  Config: rho={best_sam['rho']}, sigma_min={best_sam['sigma_min']}, sigma_max={best_sam['sigma_max']}")
        
        if 'fid' in df.columns:
            best_fid = df.loc[df['fid'].idxmin()]
            print(f"\nBest FID: {best_fid['fid']:.4f}")
            print(f"  Config: rho={best_fid['rho']}, sigma_min={best_fid['sigma_min']}, sigma_max={best_fid['sigma_max']}")


def main():
    parser = argparse.ArgumentParser(description="GEWDiff Ablation Study")
    parser.add_argument("--checkpoint", default="/kaggle/input/gewdiff-checkpoint/epoch_200.pth")
    parser.add_argument("--data_dir", default="/kaggle/input/wdc-dataset/data/test_wdc")
    parser.add_argument("--output", default="/kaggle/working/ablation_results.csv")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    print("="*60)
    print("GEWDiff Parameter Ablation Study")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    print(f"Device: {args.device}")
    print("="*60)
    
    tester = AblationTester(args.checkpoint, device=args.device)
    
    results_df = tester.test_parameters(
        args.data_dir,
        rho_values=[0.5, 0.6, 0.7, 0.8, 0.9],
        sigma_min_values=[0.002, 0.02, 0.05, 0.1, 0.2],
        sigma_max_values=[60, 80, 100],
    )
    
    tester.save_results(args.output)
    
    # Also save as JSON for easy parsing
    json_path = args.output.replace('.csv', '.json')
    results_df.to_json(json_path, orient='records')
    print(f"Also saved to {json_path}")


if __name__ == "__main__":
    main()
