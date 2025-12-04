from torch import optim
import torch
from torch import nn
import numpy as np
import numba as nb
import librosa
from torch.optim.lr_scheduler import LambdaLR

class WarmUpLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, total_iters: int, last_epoch: int = -1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing : float = 0.1, dim : int = -1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = num_classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def time_shift(wav: np.ndarray, sr: int, s_min: float, s_max: float) -> np.ndarray:
    start = int(np.random.uniform(sr * s_min, sr * s_max))
    if start >= 0:
        wav_time_shift = np.hstack((wav[start:], np.random.uniform(-0.001, 0.001, start)))
    else:
        wav_time_shift = np.hstack((np.random.uniform(-0.001, 0.001, -start), wav[:start]))
    return wav_time_shift


def resampling(x: np.ndarray, sr: int, r_min: float, r_max: float) -> np.ndarray:
    sr_new = sr * np.random.uniform(r_min, r_max)
    x = librosa.resample(y=x, orig_sr=sr, target_sr=sr_new)
    return x, sr_new


def spec_augment(mel_spec: np.ndarray, n_time_masks: int, time_mask_width: int, n_freq_masks: int, freq_mask_width: int):
    offset, begin = 0, 0
    for _ in range(n_time_masks):
        offset = np.random.randint(0, time_mask_width)
        begin = np.random.randint(0, mel_spec.shape[1] - offset)
        mel_spec[:, begin: begin + offset] = 0.0
    for _ in range(n_freq_masks):
        offset = np.random.randint(0, freq_mask_width)
        begin = np.random.randint(0, mel_spec.shape[0] - offset)
        mel_spec[begin: begin + offset, :] = 0.0
    return mel_spec


def get_polynomial_decay_schedule_with_warmup_and_hold(
    optimizer, num_warmup_steps, num_hold_steps, num_training_steps,
    lr_end=1e-7, power=1.0, last_epoch=-1
):
    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_hold_steps:
            return 1.0  # Hold phase
        elif current_step <= num_training_steps:
            decay_step = current_step - num_warmup_steps - num_hold_steps
            total_decay_steps = num_training_steps - num_warmup_steps - num_hold_steps
            pct_remaining = 1 - decay_step / float(max(1, total_decay_steps))
            decay = (lr_init - lr_end) * pct_remaining**power + lr_end
            return decay / lr_init
        else:
            return lr_end / lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
