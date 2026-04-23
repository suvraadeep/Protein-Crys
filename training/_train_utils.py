"""
Shared training loop utilities used by all four train_*.py scripts.
"""

import sys
import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def get_warmup_cosine_scheduler(optimizer, num_warmup_steps: int, num_total_steps: int):
    """Linear warmup then cosine annealing."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_total_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_regression(
    model: nn.Module,
    train_loader,
    val_loader,
    checkpoint_path: Path,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    max_epochs: int = 50,
    accum_steps: int = 4,
    patience: int = 10,
    loss_fn=None,
    device: str = "cpu",
):
    model.to(device)
    if loss_fn is None:
        loss_fn = nn.HuberLoss(delta=1.0)   # robust to outliers

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * max_epochs // accum_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    early_stop = EarlyStopping(patience=patience)

    global_step = 0
    for epoch in range(1, max_epochs + 1):
        # ── train ───────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y) / accum_steps
            loss.backward()
            train_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        train_loss /= len(train_loader)

        # ── val ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if early_stop.step(val_loss, model):
            print(f"Early stopping at epoch {epoch}.")
            break

    early_stop.restore_best(model)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved best model → {checkpoint_path}")
    return model


def train_classification(
    model: nn.Module,
    train_loader,
    val_loader,
    checkpoint_path: Path,
    class_weights=None,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    max_epochs: int = 50,
    accum_steps: int = 4,
    patience: int = 10,
    device: str = "cpu",
):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * max_epochs // accum_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    early_stop = EarlyStopping(patience=patience)

    global_step = 0
    for epoch in range(1, max_epochs + 1):
        # ── train ───────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y) / accum_steps
            loss.backward()
            train_loss += loss.item() * accum_steps
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

            if (i + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        train_loss /= len(train_loader)
        train_acc = correct / total

        # ── val ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += loss_fn(logits, y).item()
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += len(y)
        val_loss /= len(val_loader)
        val_acc = v_correct / v_total

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} acc={train_acc:.3f} "
              f"| val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if early_stop.step(val_loss, model):
            print(f"Early stopping at epoch {epoch}.")
            break

    early_stop.restore_best(model)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved best model → {checkpoint_path}")
    return model
