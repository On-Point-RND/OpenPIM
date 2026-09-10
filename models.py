import torch
import torch.nn as nn

from modules.data_collector import load_rx_filter_coeff


class EndFilter(nn.Module):
    def __init__(self, n_channels, out_filtration, filter_path):
        super(EndFilter, self).__init__()

        if out_filtration:
            filter_coeff = load_rx_filter_coeff(filter_path).flatten()[::-1].copy()

            wts = torch.from_numpy(filter_coeff).to(torch.complex64)
            kernel_size = wts.numel()
            wts_expand = wts.unsqueeze(0).unsqueeze(0).expand(
                n_channels, 1, kernel_size
            ).clone()
            self.end_filter = torch.nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding="same",
                groups=n_channels,
                bias=False,
            )
            self.end_filter.weight.data = wts_expand
            self.end_filter.weight.requires_grad = False

        self.out_filtration = out_filtration

    def forward(self, x):
        if self.out_filtration:
            cmplx_tensor = x[..., 0] + 1j * x[..., 1]
            cmplx_tensor = cmplx_tensor.permute(1, 0).unsqueeze(0)
            filt_cmplx = self.end_filter(cmplx_tensor)
            output = torch.stack((filt_cmplx.real, filt_cmplx.imag), dim=-1)
            x = output.squeeze(0).permute(1, 0, 2)
        return x.to(torch.float32)


class CoreModel(nn.Module):
    def __init__(
        self,
        n_channels,
        seq_len,
        tx_window,
        rx_window,
        hidden_size,
        backbone_type,
        batch_size,
        out_filtration,
        filter_path,
        aux_loss_present,
    ):
        super(CoreModel, self).__init__()
        self.output_size = 2  # PIM outputs: I & Q
        self.seq_len = seq_len
        self.tx_window = tx_window
        self.rx_window = rx_window
        self.hidden_size = hidden_size
        self.backbone_type = backbone_type
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.batch_first = True  # Force batch first
        self.bidirectional = False
        self.bias = True
        self.filter = EndFilter(n_channels, out_filtration, filter_path)
        self.out_filtration = out_filtration
        self.aux_loss_present = aux_loss_present

        if backbone_type == "mcp":
            from backbones.mcp import MultiChannelMLP
            self.backbone = MultiChannelMLP(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "wh_baseline":
            from backbones.wh_baseline import MultiChannelFIR
            self.backbone = MultiChannelFIR(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "wh_poly":
            from backbones.wh_poly import MultiChannelRegression
            self.backbone = MultiChannelRegression(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "wh_rnn":
            from backbones.wh_rnn import MultiChannelRNN
            self.backbone = MultiChannelRNN(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "wh_ttr":
            from backbones.wh_ttr import TemporalTransformer
            self.backbone = TemporalTransformer(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "wh_tctr":
            from backbones.wh_tctr import TemporalChannelTransformer
            self.backbone = TemporalChannelTransformer(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "wh_ctr":
            from backbones.wh_ctr import ChannelTransformer
            self.backbone = ChannelTransformer(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "rnn":
            from backbones.rnn import MultiChannelPureRNN
            self.backbone = MultiChannelPureRNN(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "tctr":
            from backbones.tctr import MultiChannelPureTCTR
            self.backbone = MultiChannelPureTCTR(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "wh_cnn1d":
            from backbones.wh_cnn1d import WhCnn1d
            self.backbone = WhCnn1d(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        elif backbone_type == "wh_cnn2d":
            from backbones.wh_cnn2d import WhCnn2d
            self.backbone = WhCnn2d(
                seq_len=self.seq_len,
                tx_filt_size=self.tx_window,
                rx_filt_size=self.rx_window,
                n_channels=self.n_channels,
            )

        else:
            raise ValueError(
                f"The backbone type '{self.backbone_type}' is not supported. Please add your own "
                f"backbone under ./backbones and update models.py accordingly."
            )

        # Initialize backbone parameters
        try:
            self.backbone.reset_parameters()
            print("Backbone Initialized...")
        except AttributeError:
            pass

    def get_expert_weights(self):
        if hasattr(self.backbone, "get_expert_weights"):
            return self.backbone.get_expert_weights()
        return None

    def get_expert_names(self):
        if hasattr(self.backbone, "get_expert_names"):
            return self.backbone.get_expert_names()
        return None

    def forward(self, x, h_0=None):
        device = x.device
        batch_size = x.size(0)  # NOTE: dim of x must be (batch, time, feat)/(N, T, F)

        if h_0 is None:  # Create initial hidden states if necessary
            h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        if self.aux_loss_present:
            output, aux_loss = self.backbone(x, h_0)
        else:
            output = self.backbone(x, h_0)
        filtered_output = self.filter(output)
        if self.aux_loss_present:
            return filtered_output, aux_loss
        else:
            return filtered_output

    def get_aux_loss_state(self):
        return self.aux_loss_present
