import torch
import torch.nn as nn
import torch.nn.functional as F


class IQComponentWiseLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, reduction="mean"):
        """
        Custom loss function for IQ signals.

        Args:
            alpha (float): Weight for component-wise loss.
            beta (float): Weight for amplitude loss.
            gamma (float): Weight for phase loss.
            reduction (str): 'mean' or 'sum' for loss reduction.
        """
        super(IQComponentWiseLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: Tensor of shape (B, N, 2), where B is batch size, N is sequence length,
              and the last dimension represents [I, Q].
        target: Tensor of shape (B, N, 2), same format as pred.
        """
        # Separate I and Q components
        I_pred, Q_pred = pred[..., 0], pred[..., 1]
        I_true, Q_true = target[..., 0], target[..., 1]

        # Component-wise loss
        loss_I = (I_true - I_pred) ** 2
        loss_Q = (Q_true - Q_pred) ** 2
        loss_comp = loss_I + loss_Q

        # Amplitude loss
        A_pred = torch.sqrt(I_pred**2 + Q_pred**2)
        A_true = torch.sqrt(I_true**2 + Q_true**2)
        loss_amp = (A_true - A_pred) ** 2

        # Phase loss (circular distance using sine)
        theta_pred = torch.atan2(Q_pred, I_pred)
        theta_true = torch.atan2(Q_true, I_true)
        phase_diff = torch.sin(theta_true - theta_pred)  # Circular difference
        loss_phase = phase_diff**2

        # Combine losses
        total_loss = (
            self.alpha * loss_comp + self.beta * loss_amp + self.gamma * loss_phase
        )

        # Apply reduction
        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            raise ValueError("Invalid reduction type. Use 'mean' or 'sum'.")


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, fft_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # Weight for physics term
        self.fft_weight = fft_weight  # Weight for spectral loss

    def forward(self, pred, target):
        # Time-domain MSE
        time_loss = self.mse(pred, target)

        # Frequency-domain MSE (magnitude)
        pred_fft = torch.abs(torch.fft.rfft(pred))
        target_fft = torch.abs(torch.fft.rfft(target))
        freq_loss = self.mse(pred_fft, target_fft)

        # Physics term (e.g., enforce 3rd-order nonlinearity)
        # Example: PIM power ∝ input_power^3
        total_loss = (1 - self.fft_weight) * time_loss + self.fft_weight * freq_loss
        return total_loss


class JointLoss(nn.Module):
    def __init__(
        self, alpha=0.3, fft_weight=0.5, odd_order_weight=0.2, compress_weight=0.4
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha  # Weight for physics-guided terms
        self.fft_weight = fft_weight  # Spectral loss weight
        self.odd_order_weight = odd_order_weight  # Odd-order penalty strength
        self.compress_weight = compress_weight  # Dynamic range compression weight

    def forward(self, pred, target):
        # --- Time-domain MSE ---
        time_loss = self.mse(pred, target)

        # --- Frequency-domain MSE (magnitude) ---
        pred_fft = torch.abs(torch.fft.rfft(pred))
        target_fft = torch.abs(torch.fft.rfft(target))
        freq_loss = self.mse(pred_fft, target_fft)

        # --- Odd-Order Term Penalty ---
        # Create mask to penalize even-order frequency components
        n_freq = pred_fft.size(-1)
        even_order_mask = torch.zeros(n_freq, device=pred.device)
        even_order_mask[::2] = (
            1  # Simple even-index mask (customize for your PIM frequencies)
        )

        # Penalize predicted even-order components (should be near zero)
        even_loss = torch.mean(pred_fft * even_order_mask)

        # --- Dynamic Range Compression ---
        # Compress time-domain signals
        compressed_pred = torch.log1p(torch.abs(pred))
        compressed_target = torch.log1p(torch.abs(target))
        compressed_time_loss = self.mse(compressed_pred, compressed_target)

        # Compress frequency magnitudes
        compressed_pred_fft = torch.log1p(pred_fft)
        compressed_target_fft = torch.log1p(target_fft)
        compressed_freq_loss = self.mse(compressed_pred_fft, compressed_target_fft)

        # --- Combine Losses ---
        total_loss = (
            (1 - self.fft_weight) * time_loss
            + self.fft_weight * freq_loss
            + self.alpha
            * (
                self.odd_order_weight * even_loss
                + self.compress_weight * (compressed_time_loss + compressed_freq_loss)
            )
        )
        return total_loss
