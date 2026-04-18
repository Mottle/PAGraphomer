#!/usr/bin/env python3
"""Extract best metrics from GRIT-motil-finetune and GRIT-finetune results."""

import json
import os
from pathlib import Path

# Define datasets to compare
motil_datasets = [
    "ogbg-molhiv",
    "ogbg-molbace",
    "ogbg-molbbbp",
    "ogbg-molclintox",
    "ogbg-molmuv",
    "ogbg-moltox21",
    "ogbg-molsider",
]

baseline_datasets = [
    "ogbg-molbace",
    "ogbg-molbbbp",
]


def extract_best_auc(stats_file):
    """Extract best AUC and corresponding epoch from stats.json."""
    best_auc = 0
    best_epoch = -1

    with open(stats_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("auc", 0) > best_auc:
                    best_auc = data.get("auc", 0)
                    best_epoch = data.get("epoch", -1)
            except:
                pass

    return best_epoch, best_auc


def extract_metrics_with_best_test(val_stats_file, test_stats_file):
    """Extract best val AUC, best test AUC, and their gaps."""
    # First pass: find best val AUC and its epoch
    best_val_auc = 0
    best_val_epoch = -1

    with open(val_stats_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("auc", 0) > best_val_auc:
                    best_val_auc = data.get("auc", 0)
                    best_val_epoch = data.get("epoch", -1)
            except:
                pass

    # Find test AUC at best val epoch
    test_auc_at_best_val = 0
    with open(test_stats_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("epoch") == best_val_epoch:
                    test_auc_at_best_val = data.get("auc", 0)
                    break
            except:
                pass

    # Second pass: find best test AUC
    best_test_auc = 0
    best_test_epoch = -1

    with open(test_stats_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("auc", 0) > best_test_auc:
                    best_test_auc = data.get("auc", 0)
                    best_test_epoch = data.get("epoch", -1)
            except:
                pass

    return {
        "best_val_epoch": best_val_epoch,
        "best_val_auc": best_val_auc,
        "test_at_best_val": test_auc_at_best_val,
        "best_test_epoch": best_test_epoch,
        "best_test_auc": best_test_auc,
    }


def main():
    base_dir = Path("/home/liar/ml/PAGraphomer/results")

    print("=" * 100)
    print("GRIT-MotiL vs GRIT (Baseline) Finetune Comparison")
    print("=" * 100)
    print()

    # Compare BACE and BBBP (have both)
    comparison_datasets = ["ogbg-molbace", "ogbg-molbbbp"]

    print("\n" + "=" * 100)
    print("Part 1: Direct GRIT-motil vs GRIT Baseline Comparison (BACE, BBBP)")
    print("=" * 100)
    print(f"{'Dataset':<20} {'Method':<15} {'Best Val Epoch':<15} {'Best Val AUC':<15}")
    print("-" * 100)

    for ds in comparison_datasets:
        # MotiL
        motil_dir = base_dir / f"{ds}-GRIT-motil-finetune" / "agg" / "val"
        if (motil_dir / "stats.json").exists():
            epoch, auc = extract_best_auc(motil_dir / "stats.json")
            print(f"{ds:<20} {'MotiL':<15} {epoch:<15} {auc:.5f}")
        else:
            print(f"{ds:<20} {'MotiL':<15} {'N/A':<15} {'N/A':<15}")

        # Baseline
        baseline_dir = base_dir / f"{ds}-GRIT-finetune" / "agg" / "val"
        if (baseline_dir / "stats.json").exists():
            epoch, auc = extract_best_auc(baseline_dir / "stats.json")
            print(f"{ds:<20} {'Baseline':<15} {epoch:<15} {auc:.5f}")
        else:
            print(f"{ds:<20} {'Baseline':<15} {'N/A':<15} {'N/A':<15}")

        print("-" * 100)

    print("\n" + "=" * 120)
    print("Part 2: GRIT-MotiL Finetune - Best Val vs Best Test Comparison")
    print("=" * 120)
    print(
        f"{'Dataset':<18} {'Val Epoch':<10} {'Val AUC':<10} {'Test@Val':<10} {'Test Epoch':<12} {'Best Test':<10} {'Gap':<10} {'备注':<15}"
    )
    print("-" * 120)

    for ds in motil_datasets:
        val_file = base_dir / f"{ds}-GRIT-motil-finetune" / "agg" / "val" / "stats.json"
        test_file = (
            base_dir / f"{ds}-GRIT-motil-finetune" / "agg" / "test" / "stats.json"
        )

        if val_file.exists() and test_file.exists():
            metrics = extract_metrics_with_best_test(val_file, test_file)

            # Calculate gap between test@val and best test
            gap = metrics["best_test_auc"] - metrics["test_at_best_val"]

            # Determine note based on gap
            if gap > 0.03:
                note = "严重滞后"
            elif gap > 0.015:
                note = "明显滞后"
            elif gap > 0.005:
                note = "轻微滞后"
            elif gap < -0.01:
                note = "test>val异常"
            else:
                note = "基本一致"

            print(
                f"{ds:<18} {metrics['best_val_epoch']:<10} {metrics['best_val_auc']:.5f}  "
                f"{metrics['test_at_best_val']:.5f}   {metrics['best_test_epoch']:<12} "
                f"{metrics['best_test_auc']:.5f}  {gap:+.5f}  {note:<15}"
            )
        else:
            print(
                f"{ds:<18} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<15}"
            )

    print("\n" + "=" * 120)
    print("Part 3: GRIT Baseline Finetune - Best Val vs Best Test Comparison")
    print("=" * 120)
    print(
        f"{'Dataset':<18} {'Val Epoch':<10} {'Val AUC':<10} {'Test@Val':<10} {'Test Epoch':<12} {'Best Test':<10} {'Gap':<10} {'备注':<15}"
    )
    print("-" * 120)

    for ds in baseline_datasets:
        val_file = base_dir / f"{ds}-GRIT-finetune" / "agg" / "val" / "stats.json"
        test_file = base_dir / f"{ds}-GRIT-finetune" / "agg" / "test" / "stats.json"

        if val_file.exists() and test_file.exists():
            metrics = extract_metrics_with_best_test(val_file, test_file)

            # Calculate gap between test@val and best test
            gap = metrics["best_test_auc"] - metrics["test_at_best_val"]

            # Determine note based on gap
            if gap > 0.03:
                note = "严重滞后"
            elif gap > 0.015:
                note = "明显滞后"
            elif gap > 0.005:
                note = "轻微滞后"
            elif gap < -0.01:
                note = "test>val异常"
            else:
                note = "基本一致"

            print(
                f"{ds:<18} {metrics['best_val_epoch']:<10} {metrics['best_val_auc']:.5f}  "
                f"{metrics['test_at_best_val']:.5f}   {metrics['best_test_epoch']:<12} "
                f"{metrics['best_test_auc']:.5f}  {gap:+.5f}  {note:<15}"
            )
        else:
            print(
                f"{ds:<18} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<15}"
            )

    print("\n" + "=" * 120)
    print("\n说明：")
    print("- 'Val Epoch' / 'Val AUC': validation set 上 AUC 最高的 epoch 及其 AUC 值")
    print("- 'Test@Val': validation 最优时的 test AUC（模型选择依据）")
    print("- 'Test Epoch' / 'Best Test': 整个训练过程中 test AUC 最高的 epoch 及其值")
    print("- 'Gap': Best Test - Test@Val，正值表示模型选择错过了更好的 test 表现")
    print("- 如果 Gap 很大，说明 val set 与 test set 分布不一致，或存在过拟合")


if __name__ == "__main__":
    main()
