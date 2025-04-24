__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import torch
import torch.nn as nn
from backbones.rvtdcnn import RVTDCNN
from scipy.signal import firwin2


class EndFilter(nn.Module):
    def __init__(self, n_channels, out_filtration):
        super(EndFilter, self).__init__()

        if out_filtration:
            Fs = 245.76e6  # Sampling frequency
            freq = [
                0,  # Start of passband
                35.08e6,  # Start of first stopband (aliased 1950 MHz)
                35.08e6,  # End of first stopband start
                99.68e6,  # Start of passband
                99.68e6,  # End of passband start
                Fs / 2,  # Nyquist frequency
            ]
            gain = [1, 1, 0, 0, 0, 0]  # 1=pass, 0=stop
            # Design filter with 255 taps
            filter_coeff = firwin2(255, freq, gain, fs=Fs)
            wts = torch.from_numpy(filter_coeff).to(torch.complex64)
            wts_expand = wts.unsqueeze(0).unsqueeze(0).expand(n_channels, 1, 255)
            self.end_filter = torch.nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=255,
                padding="valid",
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
        input_size,
        hidden_size,
        num_layers,
        backbone_type,
        batch_size,
        out_filtration,
    ):
        super(CoreModel, self).__init__()
        self.output_size = 2  # PIM outputs: I & Q
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.backbone_type = backbone_type
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.batch_first = True  # Force batch first
        self.bidirectional = False
        self.bias = True
        self.filter = EndFilter(n_channels, out_filtration)
        self.out_filtration = out_filtration

        if backbone_type == "gmp":
            from backbones.gmp import GMP

            self.backbone = GMP()
        elif backbone_type == "gru":
            from backbones.gru import GRU

            self.backbone = GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                bias=self.bias,
            )

        elif backbone_type == "lstm_cnn":
            from backbones.lstm_cnn import CNN_LSTM_FIR

            self.backbone = CNN_LSTM_FIR(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                batch_first=self.batch_first,
                bias=self.bias,
                n_channels=self.n_channels,
            )

        elif backbone_type == "dgru":
            from backbones.dgru import DGRU

            self.backbone = DGRU(
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                bias=self.bias,
                input_len=input_size,
                n_channels=n_channels,
            )
        elif backbone_type == "dgruf":
            from backbones.dgru_filter import DGRU

            self.backbone = DGRU(
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                bias=self.bias,
                input_len=input_size,
                n_channels=n_channels,
            )
        elif backbone_type == "qgru_amp1":
            from backbones.qgru_amp1 import QGRU

            self.backbone = QGRU(
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                bias=self.bias,
            )
        elif backbone_type == "lstm":
            from backbones.lstm import LSTM

            self.backbone = LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                bias=self.bias,
            )
        elif backbone_type == "vdlstm":
            from backbones.vdlstm import VDLSTM

            self.backbone = VDLSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                bias=self.bias,
            )
        elif backbone_type == "rvtdcnn":
            self.backbone = RVTDCNN(fc_hid_size=hidden_size)
        elif backbone_type == "dgru_abs_only":
            from backbones.dgru_abs import DGRU_abs

            self.backbone = DGRU_abs(
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                bias=self.bias,
            )
        elif backbone_type == "linear":
            from backbones.linear import Linear

            self.backbone = Linear(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )

        elif backbone_type == "extlinear":
            from backbones.linear_external import Linear

            self.backbone = Linear(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )
        elif backbone_type == "intlinear":
            from backbones.linear_internal import Linear

            self.backbone = Linear(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )

        elif backbone_type == "leaklinear":
            from backbones.linear_leakage import Linear

            self.backbone = Linear(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )
        elif backbone_type == "leak_int_linear":
            from backbones.linear_leak_int import Linear

            self.backbone = Linear(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )

        elif backbone_type == "simple_dimple":
            from backbones.simple_dimple import Simple

            self.backbone = Simple(
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                bias=self.bias,
                input_len=input_size,
                n_channels=n_channels,
            )

        elif backbone_type == "linpoly":
            from backbones.lin_poly import Linear

            self.backbone = Linear(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )

        elif backbone_type == "linpoly_external":
            from backbones.lin_poly_external import Linear

            self.backbone = Linear(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )

        elif backbone_type == "convx":
            from backbones.conv import ConvModel

            self.backbone = ConvModel(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )

        elif backbone_type == "mix":
            from backbones.mix import Mix

            self.backbone = Mix(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
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

    def forward(self, x, h_0=None):
        device = x.device
        batch_size = x.size(0)  # NOTE: dim of x must be (batch, time, feat)/(N, T, F)

        if h_0 is None:  # Create initial hidden states if necessary
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # Forward Propagate through the RNN
        # print('x.shape: ', x.shape)
        output = self.backbone(x, h_0)
        filtered_output = self.filter(output)
        return filtered_output


class CascadedModel(nn.Module):
    def __init__(self, pim_model):
        super(CascadedModel, self).__init__()
        self.pim_model = pim_model

    def freeze_pim_model(self):
        for param in self.pim_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.pim_model(x)
        return x
