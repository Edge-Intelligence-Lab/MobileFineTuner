#!/usr/bin/env python3
"""
Parse training log and plot loss curve
Usage: python plot_loss_curve.py <log_file> [output_image]
"""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_path):
    """Parse training log, extract step and loss"""
    steps = []
    losses = []
    
    # Match patterns: [Step X] Loss=Y or step X/... | loss Y
    pattern1 = r'\[Step (\d+)\] Loss=([\d.]+)'
    pattern2 = r'step (\d+)/\d+.*loss ([\d.]+)'
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern1, line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                continue
            
            match = re.search(pattern2, line, re.IGNORECASE)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    
    return steps, losses

def plot_loss_curve(steps, losses, output_path, title="Training Loss Curve"):
    """Plot loss curve"""
    plt.figure(figsize=(12, 6))
    
    # Main curve
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, 'b-', alpha=0.7, linewidth=0.8, label='Loss')
    
    # Add moving average line
    window = min(20, len(losses) // 5) if len(losses) > 20 else 5
    if len(losses) > window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
        plt.plot(smooth_steps, smoothed, 'r-', linewidth=2, label=f'Moving Avg (w={window})')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale
    plt.subplot(1, 2, 2)
    plt.plot(steps, losses, 'b-', alpha=0.7, linewidth=0.8)
    plt.xlabel('Step')
    plt.ylabel('Loss (log scale)')
    plt.title(f'{title} (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[DONE] Loss curve saved to: {output_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total steps: {len(steps)}")
    print(f"  Initial Loss: {losses[0]:.4f}")
    print(f"  Final Loss: {losses[-1]:.4f}")
    print(f"  Minimum Loss: {min(losses):.4f} (Step {steps[losses.index(min(losses))]})")
    print(f"  Average Loss (last 20 steps): {np.mean(losses[-20:]):.4f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_loss_curve.py <log_file> [output_image]")
        print("Example: python plot_loss_curve.py runs/gemma_1b_lora_short_300steps/train.log")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    # Default output path
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        import os
        base_dir = os.path.dirname(log_path)
        output_path = os.path.join(base_dir, "loss_curve.png")
    
    # Parse log
    steps, losses = parse_log(log_path)
    
    if not steps:
        print(f"[ERROR] Failed to parse loss data from log: {log_path}")
        sys.exit(1)
    
    print(f"[INFO] Parsed {len(steps)} data points")
    
    # Infer title from log path
    if 'gemma_1b' in log_path.lower():
        title = "Gemma 1B LoRA Training Loss"
    elif 'gpt2_medium' in log_path.lower():
        title = "GPT-2 Medium LoRA Training Loss"
    elif 'gemma' in log_path.lower():
        title = "Gemma LoRA Training Loss"
    elif 'gpt2' in log_path.lower():
        title = "GPT-2 LoRA Training Loss"
    else:
        title = "Training Loss Curve"
    
    # Plot curve
    plot_loss_curve(steps, losses, output_path, title)

if __name__ == "__main__":
    main()

