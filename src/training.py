import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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

NUM_BATCHES_PER_EPOCH = 500
MAX_GRAD_NORM = 1.0


class Trainer:
    def __init__(self, model, device="cuda", lr=0.0005, weight_decay=0.9,
                 use_ctl=False, run_name=None):
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
        self._last_grad_norm = 0.0
        self._epoch_start = 0.0
        self.scaler = torch.amp.GradScaler(device=device) if device == "cuda" else None

        self.writer = None
        if run_name is not None:
            tb_dir = os.path.join(LOGS_DIR, "tb")
            self.writer = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))
            try:
                dummy = torch.randn(1, 96, 125, device=device)
                self.writer.add_graph(self.model, dummy)
            except Exception:
                pass
            hp = {
                "lr": lr,
                "weight_decay": weight_decay,
                "use_ctl": use_ctl,
                "batch_size": 128,
                "chunk_duration": 24.0,
            }
            self.writer.add_hparams(hp, {})

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

        use_amp = self.scaler is not None

        n_total = min(len(loader), NUM_BATCHES_PER_EPOCH)
        pbar = tqdm(total=n_total, desc=f"Epoch {self.epoch}", leave=False)
        for i, (feats, boundaries, funcs, tokens) in enumerate(loader):
            if i >= NUM_BATCHES_PER_EPOCH:
                break

            feats = feats.to(self.device)
            boundaries = boundaries.to(self.device)
            funcs = funcs.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device, enabled=use_amp):
                out = self.model(feats)
                loss, extras = self._compute_loss(out, boundaries, funcs, tokens)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), MAX_GRAD_NORM
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self._last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), MAX_GRAD_NORM
                ).item()
                self.optimizer.step()

            for k, v in extras.items():
                metrics_acc[k] = metrics_acc.get(k, 0) + v
            metrics_acc["loss"] = metrics_acc.get("loss", 0) + loss.item()
            n_batches += 1

            pbar.set_postfix({k: f"{v:.4f}" for k, v in extras.items()})
            pbar.update(1)

        pbar.close()
        return {k: v / n_batches for k, v in metrics_acc.items()}

    def _log_histograms(self, epoch):
        if self.writer is None:
            return
        for name, param in self.model.named_parameters():
            if param.numel() < 100000 and param.ndim >= 2:
                self.writer.add_histogram(f"weights/{name}", param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)

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

    def fit(self, train_loader, val_loader, max_epochs=100, ckpt_path=None):
        start_epoch = self.epoch
        for epoch in range(start_epoch, max_epochs):
            self.epoch = epoch
            self._epoch_start = time.time()
            train_metrics = self.train_epoch(train_loader)
            epoch_time = time.time() - self._epoch_start

            val_start = time.time()
            val_loss = self.validate(val_loader)
            val_time = time.time() - val_start

            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            lr = self.optimizer.param_groups[0]["lr"]
            info = " | ".join(f"{k}:{v:.4f}" for k, v in train_metrics.items())
            print(f"Epoch {epoch:3d} | {info} | val: {val_loss:.4f} | lr: {lr:.2e} | {epoch_time:.1f}s")

            if self.writer:
                self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
                self.writer.add_scalar("train/boundary_loss", train_metrics.get("b", 0), epoch)
                self.writer.add_scalar("train/function_loss", train_metrics.get("f", 0), epoch)
                if "ctl" in train_metrics:
                    self.writer.add_scalar("train/ctl_loss", train_metrics["ctl"], epoch)
                self.writer.add_scalar("val/loss", val_loss, epoch)
                self.writer.add_scalar("train/learning_rate", lr, epoch)
                self.writer.add_scalar("train/grad_norm", self._last_grad_norm, epoch)
                self.writer.add_scalar("train/epoch_time", epoch_time, epoch)
                self.writer.add_scalar("val/epoch_time", val_time, epoch)
                if epoch % 5 == 0:
                    self._log_histograms(epoch)
                mem_mb = 0
                if self.device == "cuda" and torch.cuda.is_available():
                    mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                elif self.device == "mps" and torch.backends.mps.is_available():
                    try:
                        mem_mb = torch.mps.current_allocated_memory() / 1024 / 1024
                    except Exception:
                        pass
                if mem_mb > 0 and self.writer:
                    self.writer.add_scalar("system/device_allocated_mb", mem_mb, epoch)
                self.writer.flush()

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_path = os.path.join(MODELS_DIR, "best_model.pt")
                self.save_checkpoint(best_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if ckpt_path:
                self.save_checkpoint(ckpt_path)

        final_path = os.path.join(MODELS_DIR, "final_model.pt")
        self.save_checkpoint(final_path)

        if self.writer:
            self.writer.close()
        return self.train_losses, self.val_losses
