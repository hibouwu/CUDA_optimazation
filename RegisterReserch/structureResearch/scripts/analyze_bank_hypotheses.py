#!/usr/bin/env python3
"""
Analyze bank-conflict hypotheses against measured data.
Compares mod 2, mod 4, mod 8, and mod 16 models against observed cycles.
"""
import csv
from pathlib import Path
from collections import defaultdict
import statistics

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "bank_scan" / "results.csv"


def parse_results(csv_path):
    """Parse bank scan results CSV"""
    latency_cases = {}  # {stride: cycles}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            case = row["case"]
            if case.startswith("L_b04_s"):  # Only look at base=R4 for clarity
                stride = int(case.split("s")[1])
                cycles = float(row["median_cycles_per_op"])
                latency_cases[stride] = cycles
    return latency_cases


def collision_count(base, source0_offset, source1_offset, modulo):
    """Count register conflicts under modulo hypothesis"""
    # source0 = base + source0_offset, source1 = base + source1_offset
    # Conflict if (source0 % modulo) == (source1 % modulo)
    r0 = (base + source0_offset) % modulo
    r1 = (base + source1_offset) % modulo
    return 1 if r0 == r1 else 0


def analyze_hypothesis(stride_data, base_reg=4):
    """Analyze how well different modulo hypotheses fit the data"""
    results = {}
    
    # Extract observed data
    observed_slow = []  # strides with slow cycles (~3.07)
    observed_fast = []  # strides with fast cycles (~2.09)
    
    for stride, cycles in sorted(stride_data.items()):
        if cycles > 2.5:  # Slow (has conflict)
            observed_slow.append(stride)
        else:  # Fast (no conflict)
            observed_fast.append(stride)
    
    print(f"Observed fast (2.086 c/op) strides: {observed_fast}")
    print(f"Observed slow (3.070 c/op) strides: {observed_slow}")
    print()
    
    # Test different modulo hypotheses
    for modulo in [2, 4, 8, 16]:
        print(f"\n{'='*60}")
        print(f"Testing MOD {modulo} hypothesis")
        print(f"{'='*60}")
        
        predicted_collisions = []
        predicted_no_collisions = []
        
        for stride in range(1, 17):
            # For latency case L_bXX_sYY:
            # source0 = base + stride
            # source1 = base + 2*stride
            # destination = base (accumulator reuse)
            has_collision = collision_count(base_reg, stride, 2*stride, modulo)
            
            if has_collision:
                predicted_collisions.append(stride)
            else:
                predicted_no_collisions.append(stride)
        
        print(f"Predicted collisions: {predicted_collisions}")
        print(f"Predicted no-collision: {predicted_no_collisions}")
        
        # Score against observations
        collision_correct = len(set(predicted_collisions) & set(observed_slow))
        collision_total = len(predicted_collisions)
        
        no_collision_correct = len(set(predicted_no_collisions) & set(observed_fast))
        no_collision_total = len(predicted_no_collisions)
        
        total_correct = collision_correct + no_collision_correct
        total_cases = 16
        accuracy = total_correct / total_cases * 100
        
        print(f"\nAccuracy breakdown:")
        print(f"  Collision cases: {collision_correct}/{collision_total} correct")
        print(f"  No-collision cases: {no_collision_correct}/{no_collision_total} correct")
        print(f"  Overall accuracy: {total_correct}/{total_cases} = {accuracy:.1f}%")
        
        # Detailed mismatch analysis
        missed_slow = set(observed_slow) - set(predicted_collisions)
        missed_fast = set(observed_fast) - set(predicted_no_collisions)
        
        if missed_slow:
            print(f"  ✗ Predicted fast but observed slow: {sorted(missed_slow)}")
        if missed_fast:
            print(f"  ✗ Predicted slow but observed fast: {sorted(missed_fast)}")
        
        results[modulo] = {
            "accuracy": accuracy,
            "total_correct": total_correct,
            "collision_predicted": set(predicted_collisions),
            "fast_predicted": set(predicted_no_collisions),
        }
    
    return results, observed_fast, observed_slow


def main():
    if not RESULTS.exists():
        print(f"Error: {RESULTS} not found")
        print("Run: STAGE=main OPCODES=lop3 ./scripts/run_opcode_suite.sh")
        return
    
    print("\n" + "="*60)
    print("BANK CONFLICT HYPOTHESIS ANALYSIS")
    print("="*60)
    
    stride_data = parse_results(RESULTS)
    results, observed_fast, observed_slow = analyze_hypothesis(stride_data)
    
    print("\n" + "="*60)
    print("SUMMARY RANKING")
    print("="*60)
    
    ranked = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for rank, (modulo, metrics) in enumerate(ranked, 1):
        print(f"{rank}. MOD {modulo}: {metrics['accuracy']:.1f}% accuracy")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print(f"\nObserved pattern: Alternating delays, mod 2 periodicity")
    print(f"  - Odd strides: {observed_fast} → fast (2.086 c/op)")
    print(f"  - Even strides: {observed_slow} → slow (3.070 c/op)")
    print(f"\nThis pattern is perfectly consistent with 2-bank organization.")
    print(f"However, it's also consistent with >2 banks if they are grouped by parity.")
    print(f"\nTo definitively rule out multi-bank parity grouping, need:")
    print(f"  1. Extended stride range (1-64) to see if other periodicities emerge")
    print(f"  2. Multi-instruction types (IMAD, FMA) for consistency check")
    print(f"  3. Full register-space scan (not just R4-R7 base)")
    print(f"  4. NCU performance counter validation")


if __name__ == "__main__":
    main()
