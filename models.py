__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import torch
import torch.nn as nn
from backbones.rvtdcnn import RVTDCNN


class CoreModel(nn.Module):
    def __init__(self, n_channels, input_size, hidden_size, num_layers, backbone_type, batch_size):
        super(CoreModel, self).__init__()
        self.output_size = 2  # PIM outputs: I & Q
        self.n_channels = n_channels
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.backbone_type = backbone_type
        self.batch_size = batch_size
        self.batch_first = True  # Force batch first
        self.bidirectional = False
        self.bias = True

        if backbone_type == 'gmp':
            from backbones.gmp import GMP
            self.backbone = GMP()
        elif backbone_type == 'gru':
            from backbones.gru import GRU
            self.backbone = GRU(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                output_size=self.output_size,
                                num_layers=self.num_layers,
                                batch_size=self.batch_size,
                                bidirectional=self.bidirectional,
                                batch_first=self.batch_first,
                                bias=self.bias)
        elif backbone_type == 'dgru':
            from backbones.dgru import DGRU
            self.backbone = DGRU(hidden_size=self.hidden_size,
                                 output_size=self.output_size,
                                 num_layers=self.num_layers,
                                 batch_size=self.batch_size,
                                 bidirectional=self.bidirectional,
                                 batch_first=self.batch_first,
                                 bias=self.bias)
        elif backbone_type == 'qgru':
            from backbones.qgru import QGRU
            self.backbone = QGRU(hidden_size=self.hidden_size,
                                 output_size=self.output_size,
                                 num_layers=self.num_layers,
                                 batch_size=self.batch_size,
                                 bidirectional=self.bidirectional,
                                 batch_first=self.batch_first,
                                 bias=self.bias)
        elif backbone_type == 'qgru_amp1':
            from backbones.qgru_amp1 import QGRU
            self.backbone = QGRU(hidden_size=self.hidden_size,
                                 output_size=self.output_size,
                                 num_layers=self.num_layers,
                                 batch_size=self.batch_size,
                                 bidirectional=self.bidirectional,
                                 batch_first=self.batch_first,
                                 bias=self.bias)
        elif backbone_type == 'lstm':
            from backbones.lstm import LSTM
            self.backbone = LSTM(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 output_size=self.output_size,
                                 num_layers=self.num_layers,
                                 batch_size=self.batch_size,
                                 bidirectional=self.bidirectional,
                                 batch_first=self.batch_first,
                                 bias=self.bias)
        elif backbone_type == 'vdlstm':
            from backbones.vdlstm import VDLSTM
            self.backbone = VDLSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   output_size=self.output_size,
                                   num_layers=self.num_layers,
                                   batch_size=self.batch_size,
                                   bidirectional=self.bidirectional,
                                   batch_first=self.batch_first,
                                   bias=self.bias)
        elif backbone_type == 'rvtdcnn':
            self.backbone = RVTDCNN(fc_hid_size=hidden_size)
        elif backbone_type == 'dgru_abs_only':
            from backbones.dgru_abs import DGRU_abs
            self.backbone = DGRU_abs(hidden_size=self.hidden_size,
                                 output_size=self.output_size,
                                 num_layers=self.num_layers,
                                 batch_size=self.batch_size,
                                 bidirectional=self.bidirectional,
                                 batch_first=self.batch_first,
                                 bias=self.bias)
        elif backbone_type == 'linear':
            from backbones.linear import Linear
            self.backbone = Linear(n_channels = n_channels,
                                 input_size = self.input_size,
                                 output_size=self.output_size,
                                 batch_size=self.batch_size)
        else:
            raise ValueError(f"The backbone type '{self.backbone_type}' is not supported. Please add your own "
                             f"backbone under ./backbones and update models.py accordingly.")

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
        out = self.backbone(x, h_0)

        return out


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
