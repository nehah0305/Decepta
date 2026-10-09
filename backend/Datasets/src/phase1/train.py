import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import CACHE_ROOT, CHECKPOINT_ROOT, DATA_ROOT, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, FRAME_SIZE, FRAMES_PER_VIDEO, METADATA_ROOT
from .model import FrameCNN
from .video_dataset import VideoDataset


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, training):
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(videos)
                loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 1 video classifier.")
    parser.add_argument("--manifest", type=Path, default=METADATA_ROOT / "dfd.csv")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--frames", type=int, default=FRAMES_PER_VIDEO)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = VideoDataset(args.manifest, DATA_ROOT, "train", args.frames, FRAME_SIZE, args.cache_dir)
    validation_set = VideoDataset(args.manifest, DATA_ROOT, "validation", args.frames, FRAME_SIZE, args.cache_dir)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, **loader_options)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = FrameCNN(args.frames).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0

    print(f"Using device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, True)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, loss_fn, optimizer, scaler, device, False)
        print(f"Epoch {epoch}/{args.epochs} | train loss={train_loss:.4f} acc={train_accuracy:.4f} | validation loss={validation_loss:.4f} acc={validation_accuracy:.4f}")
        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_accuracy": best_accuracy}, CHECKPOINT_ROOT / "best_frame_cnn.pt")


if __name__ == "__main__":
    main()