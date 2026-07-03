#!/usr/bin/env python3
"""
多 base 寄存器一致性分析

验证 2-bank 架构是否在所有寄存器上一致：
- 测试 base 寄存器 R4, R5, R6, R7
- 对比它们的 stride 1-16 碰撞模式
- 验证 mod 2 映射的一致性
"""

import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "bank_scan" / "results.csv"


def predict_collision(base_reg, source0_offset, source1_offset):
    """Predict collision under mod 2 hypothesis"""
    reg0 = (base_reg + source0_offset) % 2
    reg1 = (base_reg + source1_offset) % 2
    return reg0 == reg1


def analyze_cross_register():
    """Analyze collision patterns across all base registers"""
    
    # Read data
    data_by_base = defaultdict(dict)
    with open(RESULTS, 'r') as f:
        for row in csv.DictReader(f):
            case = row['case']
            if case.startswith('L_b'):
                # Parse: L_b04_s01 → base=4, stride=1
                parts = case.split('_')
                base = int(parts[1][1:])
                stride = int(parts[2][1:])
                cycles = float(row['median_cycles_per_op'])
                data_by_base[base][stride] = cycles
    
    print("=" * 80)
    print("CROSS-REGISTER BANK CONSISTENCY ANALYSIS")
    print("=" * 80)
    print()
    
    # Collision detection threshold
    THRESHOLD = 2.5
    
    # Analyze each base register
    bases = sorted(data_by_base.keys())
    patterns = {}
    
    for base in bases:
        strides = sorted(data_by_base[base].keys())
        pattern = []
        collision_count = 0
        
        print(f"Base Register R{base}:")
        print("-" * 40)
        
        for stride in strides:
            cycles = data_by_base[base][stride]
            is_collision = cycles > THRESHOLD
            pattern.append(is_collision)
            collision_count += is_collision
            
            marker = "⚡" if is_collision else "✓"
            parity = f"[{('even' if stride % 2 == 0 else 'odd')}]"
            print(f"  stride {stride:2d} {parity:8s}: {cycles:.6f} c/op {marker}")
        
        patterns[base] = pattern
        print(f"  → Collisions: {collision_count}/16 = {100*collision_count/16:.1f}%")
        print()
    
    # Cross-register consistency check
    print("=" * 80)
    print("CONSISTENCY VERIFICATION")
    print("=" * 80)
    print()
    
    # Check if all bases show identical pattern
    reference_pattern = patterns[bases[0]]
    all_consistent = all(patterns[b] == reference_pattern for b in bases[1:])
    
    if all_consistent:
        print("✓ ALL BASE REGISTERS SHOW IDENTICAL COLLISION PATTERN")
        print()
        print("Pattern summary (mod 2):")
        for i, (stride, collision) in enumerate(zip(range(1, 17), reference_pattern)):
            status = "collision" if collision else "no collision"
            parity = stride % 2
            print(f"  stride {stride:2d} (mod 2 = {parity}): {status:12s} → R(base+{stride}) mod 2 = ?")
        print()
        
        # Deduce mod 2 mapping
        print("Deduced mod 2 mapping:")
        print("  Odd strides (1,3,5,...,15) → no collision")
        print("  Even strides (2,4,6,...,16) → collision")
        print()
        print("This is consistent with: (base + even_offset) mod 2 = collision")
        print("Or equivalently: offset_parity matters for collision")
        print()
    else:
        print("✗ INCONSISTENT PATTERNS ACROSS BASE REGISTERS")
        for base in bases:
            print(f"  R{base}: {patterns[base]}")
    
    # Mod 2 accuracy across all bases and strides
    print("=" * 80)
    print("MOD 2 HYPOTHESIS ACCURACY (ALL BASES)")
    print("=" * 80)
    print()
    
    total_predictions = 0
    correct_predictions = 0
    
    for base in bases:
        for stride in range(1, 17):
            # Predict collision for this (base, stride) pair
            # LOP3 uses: Rbase (dest), R(base+stride) (src0), R(base+2*stride) (src1)
            # Collision occurs if both sources map to same bank: (base+stride) mod 2 == (base+2*stride) mod 2
            # This simplifies to: stride mod 2 == 0 (since 2*stride mod 2 = 0 always)
            
            predicted_collision = (stride % 2) == 0
            actual_cycles = data_by_base[base][stride]
            actual_collision = actual_cycles > THRESHOLD
            
            total_predictions += 1
            if predicted_collision == actual_collision:
                correct_predictions += 1
    
    accuracy = 100 * correct_predictions / total_predictions
    print(f"Predicted collisions (stride % 2 == 0): mod 2 hypothesis")
    print(f"Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1f}%")
    print()
    
    if accuracy == 100.0:
        print("✓ MOD 2 HYPOTHESIS PERFECTLY FITS ALL DATA")
        print("  This strongly suggests 2-bank register file organization")
        print("  with bank assignment: register_id % 2 → bank_id")
    
    return all_consistent and accuracy == 100.0


if __name__ == "__main__":
    result = analyze_cross_register()
    print()
    print("=" * 80)
    if result:
        print("CONCLUSION: 2-bank architecture confirmed across all tested registers")
    else:
        print("CONCLUSION: Further investigation needed")
    print("=" * 80)
