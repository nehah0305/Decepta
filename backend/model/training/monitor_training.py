"""
Training Progress Monitor for Decepta.
Reads the training metrics CSV log and reports progress at equal intervals.
Usage: python monitor_training.py --log <path_to_metrics.csv> --interval 60
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path


def read_latest_metrics(log_path: str) -> dict:
    """Read the last row from the metrics CSV log."""
    if not os.path.exists(log_path):
        return None
    
    last_row = None
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    
    return last_row


def format_progress(row: dict, total_epochs: int) -> str:
    """Format a progress summary from the metrics row."""
    if row is None:
        return "No training data available yet..."
    
    epoch = int(row.get("epoch", 0))
    progress_pct = epoch / total_epochs * 100 if total_epochs > 0 else 0
    
    lines = [
        f"=" * 70,
        f"TRAINING PROGRESS UPDATE - Epoch {epoch}/{total_epochs} ({progress_pct:.0f}%)",
        f"=" * 70,
        f"  Train Loss:     {float(row.get('train_loss', 0)):.4f}",
        f"  Val Loss:       {float(row.get('val_loss', 0)):.4f}",
        f"  Val AUC:        {float(row.get('val_auc', 0.5)) * 100:.2f}%",
        f"  Val Balanced:   {float(row.get('val_balanced_acc', 0)) * 100:.2f}%",
        f"  Val Recall:     {float(row.get('val_recall', 0)) * 100:.2f}%",
        f"  Val F1:         {float(row.get('val_f1', 0)) * 100:.2f}%",
        f"  Learning Rate:  {float(row.get('lr', 0)):.2e}",
        f"  Epoch Time:     {float(row.get('epoch_time_s', 0)):.1f}s",
        f"=" * 70,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monitor Decepta training progress")
    parser.add_argument("--log", type=str, required=True, help="Path to metrics CSV log")
    parser.add_argument("--interval", type=int, default=300, help="Reporting interval in seconds")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    args = parser.parse_args()
    
    print(f"Monitoring training log: {args.log}")
    print(f"Reporting every {args.interval}s ({args.interval // 60}min)")
    print(f"Press Ctrl+C to stop monitoring.\n")
    
    last_reported_epoch = -1
    
    while True:
        row = read_latest_metrics(args.log)
        if row is not None:
            current_epoch = int(row.get("epoch", 0))
            if current_epoch != last_reported_epoch:
                print(format_progress(row, args.epochs))
                last_reported_epoch = current_epoch
                
                if current_epoch >= args.epochs:
                    print("\nTraining complete!")
                    break
        else:
            print("Waiting for training to start...")
        
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
