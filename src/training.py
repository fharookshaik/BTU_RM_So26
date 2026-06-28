import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ctl_loss import ctl_loss_batch

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

BOUNDARY_LOSS_WEIGHT = 0.9
FUNCTION_LOSS_WEIGHT = 0.1
CTL_LOSS_WEIGHT = 0.1


class Trainer:
    def __init__(self, model, device="mps", lr=0.0005, weight_decay=0.9, use_ctl=False):
        self.model = model.to(device)
        self.device = device
        self.use_ctl = use_ctl
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.5
        )
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.max_patience = 4
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []

    def save_checkpoint(self, path):
        torch.save({
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["val_losses"]
        self.epoch = ckpt["epoch"]

    def _compute_loss(self, out, boundaries, funcs, tokens):
        if out.dim() == 2:
            T = boundaries.shape[1]
            center = T // 2
            b_target = boundaries[:, center]
            f_target = funcs[:, center, :]
            bce_b = F.binary_cross_entropy_with_logits(out[:, 0], b_target)
            bce_f = F.binary_cross_entropy_with_logits(out[:, 1:], f_target)
            loss = BOUNDARY_LOSS_WEIGHT * bce_b + FUNCTION_LOSS_WEIGHT * bce_f
            extras = {"b": bce_b.item(), "f": bce_f.item()}
        else:
            b_target = boundaries
            f_target = funcs
            bce_b = F.binary_cross_entropy_with_logits(out[:, :, 0], b_target)
            bce_f = F.binary_cross_entropy_with_logits(out[:, :, 1:], f_target)
            loss = BOUNDARY_LOSS_WEIGHT * bce_b + FUNCTION_LOSS_WEIGHT * bce_f
            extras = {"b": bce_b.item(), "f": bce_f.item()}
            if self.use_ctl and tokens is not None and any(len(t) > 0 for t in tokens):
                padded, lengths = self._pad_tokens(tokens)
                ctl = ctl_loss_batch(out[:, :, 1:], padded, lengths)
                loss = loss + CTL_LOSS_WEIGHT * ctl
                extras["ctl"] = ctl.item()
        return loss, extras

    def _pad_tokens(self, tokens):
        max_len = max(len(t) for t in tokens) if tokens else 1
        if max_len == 0:
            max_len = 1
        padded = torch.zeros((len(tokens), max_len), dtype=torch.long, device=self.device)
        lengths = torch.zeros(len(tokens), dtype=torch.long, device=self.device)
        for i, t in enumerate(tokens):
            tl = t.to(self.device) if isinstance(t, torch.Tensor) else t
            padded[i, :len(tl)] = tl
            lengths[i] = len(tl)
        return padded, lengths

    def train_epoch(self, loader):
        self.model.train()
        metrics_acc = {}
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {self.epoch}", leave=False)
        for feats, boundaries, funcs, tokens in pbar:
            feats = feats.to(self.device)
            boundaries = boundaries.to(self.device)
            funcs = funcs.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(feats)
            loss, extras = self._compute_loss(out, boundaries, funcs, tokens)
            loss.backward()
            self.optimizer.step()

            for k, v in extras.items():
                metrics_acc[k] = metrics_acc.get(k, 0) + v
            metrics_acc["loss"] = metrics_acc.get("loss", 0) + loss.item()
            n_batches += 1

            pbar.set_postfix({k: f"{v:.4f}" for k, v in extras.items()})

        return {k: v / n_batches for k, v in metrics_acc.items()}

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for feats, boundaries, funcs, tokens in loader:
            feats = feats.to(self.device)
            boundaries = boundaries.to(self.device)
            funcs = funcs.to(self.device)
            out = self.model(feats)
            loss, _ = self._compute_loss(out, boundaries, funcs, tokens)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(self, train_loader, val_loader, max_epochs=100):
        for epoch in range(max_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            lr = self.optimizer.param_groups[0]["lr"]
            info = " | ".join(f"{k}:{v:.4f}" for k, v in train_metrics.items())
            print(f"Epoch {epoch:3d} | {info} | val: {val_loss:.4f} | lr: {lr:.2e}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                ckpt_path = os.path.join(MODELS_DIR, "best_model.pt")
                self.save_checkpoint(ckpt_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        ckpt_path = os.path.join(MODELS_DIR, "final_model.pt")
        self.save_checkpoint(ckpt_path)
        return self.train_losses, self.val_losses
