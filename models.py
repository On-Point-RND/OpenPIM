import torch
import torch.nn as nn
from scipy.signal import firwin2
from scipy.io import loadmat


class EndFilter(nn.Module):
    def __init__(self, n_channels, out_filtration, filter_path, filter_same):
        super(EndFilter, self).__init__()

        if out_filtration:
            if filter_same:
                filter_coeff = loadmat(filter_path)["flt_coeff"][0]
                filter_coeff = filter_coeff[::-1].copy()
            else:

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

            # по

            wts_expand = wts.unsqueeze(0).unsqueeze(0).expand(n_channels, 1, 255)
            self.end_filter = torch.nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=255,
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
        input_size,
        out_window,
        medium_sim_size,
        hidden_size,
        num_layers,
        backbone_type,
        batch_size,
        out_filtration,
        filter_path,
        filter_same,
    ):
        super(CoreModel, self).__init__()
        self.output_size = 2  # PIM outputs: I & Q
        self.input_size = input_size
        self.out_window = out_window
        self.medium_sim_size = medium_sim_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.backbone_type = backbone_type
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.batch_first = True  # Force batch first
        self.bidirectional = False
        self.bias = True
        self.filter = EndFilter(n_channels, out_filtration, filter_path, filter_same)
        self.out_filtration = out_filtration

        if backbone_type == "linear":
            from backbones.linear import Linear

            self.backbone = Linear(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )

        elif backbone_type == "cond_linear":
            from backbones.linear_conductive import LinearConductive

            self.backbone = LinearConductive(
                in_seq_size=self.input_size,
                out_seq_size=self.out_window,
                n_channels=n_channels,
            )

        elif backbone_type == "cond_linear_cx":
            from backbones.linear_conductive_cx import LinearConductive

            self.backbone = LinearConductive(
                in_seq_size=self.input_size,
                out_seq_size=self.out_window,
                n_channels=n_channels,
            )

        elif backbone_type == "leak_linear":
            from backbones.linear_leakage import LinearLeakage

            self.backbone = LinearLeakage(
                in_seq_size=self.input_size,
                out_seq_size=self.out_window,
                n_channels=n_channels,
            )

        elif backbone_type == "ext_linear":
            from backbones.linear_external import LinearExternal

            self.backbone = LinearExternal(
                in_seq_size=self.input_size,
                out_seq_size=self.out_window,
                n_channels=n_channels,
            )

        elif backbone_type == "cond_leak_linear":
            from backbones.linear_cond_leak import LinearCondLeak

            self.backbone = LinearCondLeak(
                in_seq_size=self.input_size,
                out_seq_size=self.out_window,
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
            from backbones.lin_poly import LinPoly

            self.backbone = LinPoly(
                input_size=self.input_size,
                output_size=self.output_size,
                n_channels=n_channels,
                batch_size=self.batch_size,
            )

        elif backbone_type == "ext_linpoly":
            from backbones.lin_poly_external import LinPolyExternal

            self.backbone = LinPolyExternal(
                input_size=self.input_size,
                output_size=self.output_size,
                n_channels=n_channels,
                batch_size=self.batch_size,
                out_window=self.out_window,
            )

        elif backbone_type == "leak_linpoly":
            from backbones.lin_poly_leakage import LinPolyLeakage

            self.backbone = LinPolyLeakage(
                input_size=self.input_size,
                output_size=self.output_size,
                n_channels=n_channels,
                batch_size=self.batch_size,
                out_window=self.out_window,
            )

        elif backbone_type == "int_linpoly":
            from backbones.lin_poly_internal import LinPolyInternal

            self.backbone = LinPolyInternal(
                input_size=self.input_size,
                output_size=self.output_size,
                n_channels=n_channels,
                batch_size=self.batch_size,
                out_window=self.out_window,
            )

        elif backbone_type == "convx":
            from backbones.conv import ConvModel

            self.backbone = ConvModel(
                input_size=self.input_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                n_channels=n_channels,
            )

        elif backbone_type == "cnn_gru":
            from backbones.cnn_gru import CNN_GRU

            self.backbone = CNN_GRU(
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                batch_first=self.batch_first,
                bias=self.bias,
                input_size=self.input_size,
                n_channels=n_channels,
            )

        elif backbone_type == "cnn_only":
            from backbones.cnn_gru import CNN_ONLY

            self.backbone = CNN_ONLY(
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                batch_size=self.batch_size,
                batch_first=self.batch_first,
                bias=self.bias,
                input_size=self.input_size,
                n_channels=n_channels,
            )

        elif backbone_type == "learn_nonlin":
            from backbones.dnn_non_linear import LinearConductive

            self.backbone = LinearConductive(
                in_seq_size=self.input_size,
                out_seq_size=self.out_window,
                n_channels=n_channels,
            )

        elif backbone_type == "nonlin_indy":
            from backbones.dnn_non_linear_indy import LinearConductive

            self.backbone = LinearConductive(
                in_seq_size=self.input_size,
                out_seq_size=self.out_window,
                n_channels=n_channels,
            )

        elif backbone_type == "nonlin_indy_E":
            from backbones.dnn_non_linear_indy_enhanced import LinearConductive

            self.backbone = LinearConductive(
                in_seq_size=self.input_size,
                out_seq_size=self.out_window,
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
