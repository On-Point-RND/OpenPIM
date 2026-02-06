import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv

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

    def forward(self, pred, target, model=None, iteration=0):
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
    def __init__(self, alpha=0.5, fft_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.fft_weight = fft_weight  # Weight for spectral loss (0 = time-only, 1 = freq-only)

    def forward(self, pred, target, model=None, iteration=0):
        # Time-domain MSE
        time_loss = self.mse(pred, target)

        # Frequency-domain MSE using full complex FFT (real + imaginary parts)
        pred_fft = torch.fft.rfft(pred)
        target_fft = torch.fft.rfft(target)
        
        # Compute MSE on real and imaginary components separately
        #freq_loss_real = self.mse(pred_fft, target_fft)
        freq_loss_real = self.mse(torch.real(pred_fft), torch.real(target_fft))
        freq_loss_imag = self.mse(torch.imag(pred_fft), torch.imag(target_fft))
        freq_loss = freq_loss_real + freq_loss_imag

        # Combine losses
        total_loss =  self.fft_weight * freq_loss
        return total_loss


class FFTLoss(nn.Module):
    def __init__(self, bin=1900):
        super().__init__()
        self.bin = bin

    def forward(self, pred, target, model=None, iteration=0):
        fft_pred = torch.fft.rfft(pred)
        fft_true = torch.fft.rfft(target)
        # Focus on target frequency bin
        loss = torch.mean(
            torch.abs(fft_pred[:, :, self.bin] - fft_true[:, :, self.bin]) ** 2
        )
        return loss


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

    def forward(self, pred, target, model=None):
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


class AdaptiveLoss(nn.Module):
    def __init__(self, beta=0.0001, gamma=0.0001, init_iteration=1e3, log_dir='.'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.beta = beta
        self.gamma = gamma
        self.init_iteration = init_iteration
        self._was_training = True  # Track state changes
        self.log_dir = log_dir
        self.csv_logged = False  # Ensure we log only once on eval switch

        # Prepare CSV file path
        self.csv_path = os.path.join(self.log_dir, 'adaptive_loss_log.csv')
        # Write header if file doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'time_loss', 'lambdas_loss', 'entropy_loss', 'total_loss',
                    'alphas'
                ])

    def forward(self, pred, target, model, iteration):
        # 1. Access lambdas
        param = model.backbone.nlin_layer.lambdas.squeeze()
        all_comprs = model.backbone.nlin_layer.all_comprs

        # 2. Compute Softmax (alphas)
        soft = F.softmax(param, dim=0)

        # 3. Handle Print and Logging on Eval Switch
        if not model.training:
            if self._was_training:  # Only act once per switch to eval
                print(f"\n[Eval Mode] Alphas (Softmaxed Lambdas): {soft.detach().cpu().numpy()}")

                # --- LOG TO CSV ONCE ---
                if not self.csv_logged:
                    # Recompute losses for logging (same logic as below)
                    time_loss = self.mse(pred, target).item()

                    eps = 1e-10
                    entropy_loss = torch.dot(-torch.log(soft + eps), soft).item()
                    lambdas_loss = torch.dot(all_comprs.float(), soft).item()

                    if iteration == 0:
                        total_loss = time_loss
                    elif iteration > 2 * self.init_iteration:
                        total_loss = time_loss + self.beta * lambdas_loss + self.gamma * entropy_loss
                    elif iteration > self.init_iteration:
                        total_loss = time_loss + self.beta * lambdas_loss
                    else:
                        total_loss = time_loss

                    alphas_str = ','.join([f"{a:.6f}" for a in soft.detach().cpu().numpy()])

                    with open(self.csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            int(iteration),
                            time_loss,
                            lambdas_loss,
                            entropy_loss,
                            total_loss,
                            alphas_str
                        ])

                    self.csv_logged = True  # Prevent future logging until next train→eval switch

                self._was_training = False
        else:
            self._was_training = True
            self.csv_logged = False  # Reset flag when back to training

        # 4. Standard MSE
        time_loss = self.mse(pred, target)

        # 5. Stabilized Entropy Loss
        eps = 1e-10
        entropy_loss = torch.dot(-torch.log(soft + eps), soft)

        # 6. Lambda/Compression Loss
        lambdas_loss = torch.dot(all_comprs.float(), soft)

        # 7. Adaptive Logic
        if iteration == 0:
            total_loss = time_loss
        elif iteration > 2 * self.init_iteration:
            total_loss = time_loss + self.beta * lambdas_loss + self.gamma * entropy_loss
        elif iteration > self.init_iteration:
            total_loss = time_loss + self.beta * lambdas_loss
        else:
            total_loss = time_loss

        return total_loss