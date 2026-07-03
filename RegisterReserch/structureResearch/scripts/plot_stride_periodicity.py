#!/usr/bin/env python3
"""
Visualize bank-stride scan results with periodicity analysis and extrapolation.
Shows stride 1-16 measured data and mod-N hypothesis curves up to stride 32+.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "bank_scan" / "results.csv"
OUTPUT_PATH = ROOT / "assets" / "stride_periodicity_analysis.png"


def parse_latency_data(csv_path, base_reg=4):
    """Parse stride latency data for specific base register"""
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            case = row["case"]
            if case.startswith(f"L_b{base_reg:02d}_s"):
                stride = int(case.split("_s")[1])
                cycles = float(row["median_cycles_per_op"])
                data[stride] = cycles
    return data


def predict_collision(base, source0_offset, source1_offset, modulo):
    """Predict if collision occurs under given modulo hypothesis"""
    r0 = (base + source0_offset) % modulo
    r1 = (base + source1_offset) % modulo
    return 1 if r0 == r1 else 0


def generate_hypothesis_curves(base_reg=4, max_stride=32):
    """Generate predicted collision patterns for different modulo hypotheses"""
    hypotheses = {}
    for modulo in [2, 4, 8, 16]:
        predictions = []
        for stride in range(1, max_stride + 1):
            collision = predict_collision(base_reg, stride, 2*stride, modulo)
            # Convert collision (0/1) to cycles estimate
            cycles = 3.070 if collision else 2.086
            predictions.append(cycles)
        hypotheses[modulo] = predictions
    return hypotheses


def main():
    if not RESULTS.exists():
        print(f"Error: {RESULTS} not found. Run: bash scripts/run_bank_scan.sh")
        return
    
    # Parse measured data (stride 1-16)
    measured = parse_latency_data(RESULTS, base_reg=4)
    if not measured:
        print("Error: No latency data found in results")
        return
    
    measured_strides = sorted(measured.keys())
    measured_cycles = [measured[s] for s in measured_strides]
    
    # Generate hypothesis predictions (stride 1-32)
    max_stride = 32
    hypotheses = generate_hypothesis_curves(base_reg=4, max_stride=max_stride)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ============ Left plot: Measured data + hypothesis curves ============
    stride_range = np.arange(1, max_stride + 1)
    
    # Plot measured data (stride 1-16)
    ax1.scatter(measured_strides, measured_cycles, s=100, color='red', 
               label='Measured (stride 1-16)', zorder=5, marker='o')
    
    # Plot hypothesis curves (stride 1-32)
    colors = {'2': 'blue', '4': 'green', '8': 'orange', '16': 'purple'}
    for modulo, color in colors.items():
        predictions = hypotheses[int(modulo)]
        ax1.plot(stride_range, predictions, '--', linewidth=2, 
                label=f'mod {modulo} hypothesis', color=color, alpha=0.7)
    
    # Add vertical line to separate measured/predicted
    ax1.axvline(x=16.5, color='gray', linestyle=':', linewidth=1.5, 
               label='Measured range limit')
    
    ax1.set_xlabel('Stride', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cycles per LOP3', fontsize=12, fontweight='bold')
    ax1.set_title('Bank-Stride Scan: Measured vs Hypothesis Predictions', 
                 fontsize=13, fontweight='bold')
    ax1.set_ylim([1.8, 3.3])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xticks(range(1, max_stride + 1, 2))
    
    # ============ Right plot: Collision map heatmap ============
    # Create a collision matrix: rows=modulo, cols=stride
    modulo_list = [2, 4, 8, 16]
    collision_matrix = []
    for modulo in modulo_list:
        row = []
        for stride in range(1, max_stride + 1):
            collision = predict_collision(4, stride, 2*stride, modulo)
            row.append(collision)
        collision_matrix.append(row)
    
    im = ax2.imshow(collision_matrix, cmap='RdYlBu_r', aspect='auto', 
                   interpolation='nearest')
    
    ax2.set_xlabel('Stride', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bank Organization Hypothesis', fontsize=12, fontweight='bold')
    ax2.set_title('Collision Prediction Map\n(Red=Collision, Blue=No Collision)', 
                 fontsize=13, fontweight='bold')
    
    # Set ticks
    ax2.set_xticks(range(0, max_stride, 2))
    ax2.set_xticklabels(range(1, max_stride + 1, 2))
    ax2.set_yticks(range(len(modulo_list)))
    ax2.set_yticklabels([f'mod {m}' for m in modulo_list])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Collision', fontsize=10)
    
    # Add vertical line at stride 16.5
    ax2.axvline(x=15.5, color='green', linestyle='-', linewidth=2.5, alpha=0.8)
    ax2.text(16.2, -0.7, 'Measured→', fontsize=9, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_PATH}")
    
    # Print analysis summary
    print("\n" + "="*70)
    print("PERIODICITY ANALYSIS: Stride 1-16 Measured + 1-32 Predictions")
    print("="*70)
    
    print(f"\nMeasured data (stride 1-16):")
    for stride in measured_strides:
        cycles = measured[stride]
        marker = "⚡" if cycles > 2.5 else "✓"
        print(f"  stride {stride:2d}: {cycles:.6f} c/op  {marker}")
    
    print(f"\nOdd/Even pattern:")
    odd_count = sum(1 for s in measured_strides if s % 2 == 1 and measured[s] < 2.5)
    even_count = sum(1 for s in measured_strides if s % 2 == 0 and measured[s] > 2.5)
    print(f"  Odd strides (1,3,5,...,15): {odd_count}/8 fast")
    print(f"  Even strides (2,4,6,...,16): {even_count}/8 slow")
    
    print(f"\nHypothesis accuracy on measured data (stride 1-16):")
    for modulo in [2, 4, 8, 16]:
        correct = 0
        for stride in measured_strides:
            predicted_collision = predict_collision(4, stride, 2*stride, modulo)
            actual_collision = 1 if measured[stride] > 2.5 else 0
            if predicted_collision == actual_collision:
                correct += 1
        accuracy = correct / len(measured_strides) * 100
        print(f"  mod {modulo:2d}: {correct}/16 = {accuracy:5.1f}%")
    
    print(f"\nExtrapolation to stride 1-32:")
    print(f"  If mod 2: ALL strides should show alternating 2.086 / 3.070")
    print(f"  If mod 4: stride 4,8,12,16,20,24,28,32 should show 3.070")
    print(f"           stride 1-3,5-7,9-11,... should show 2.086")
    print(f"  If mod 8: stride 8,16,24,32 should show different latency")
    print(f"  If mod 16: stride 16 should show unique latency")
    print(f"\n  ⚠ To verify: extend MAX_STRIDE in scripts/patch_bank_scan.py")
    print(f"             and re-run bash scripts/run_bank_scan.sh")


if __name__ == "__main__":
    main()
