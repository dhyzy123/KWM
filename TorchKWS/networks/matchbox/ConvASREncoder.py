from typing import List, Optional, Tuple
import torch.nn as nn
import torch
from networks.matchbox.ConvBlock import ConvBlock, conv_activations, init_weights

class ConvASREncoder(nn.Module):
    def __init__(
        self,
        activation: str = "relu",
        feat_in: int = 64,
        normalization_mode: str = "batch",
        residual_mode: str = "add",
        norm_groups: int = -1,
        conv_mask: bool = True,
        frame_splicing: int = 1,
        init_mode: Optional[str] = 'xavier_uniform'
    ):
        super(ConvASREncoder, self).__init__()
        activation = conv_activations[activation]()

        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        feat_in = feat_in * frame_splicing

        self._feat_in = feat_in

        encoder_layers = []
        self.dense_residual = False
                
        self.conv_block_1 = ConvBlock(
            feat_in,
            128,
            repeat=1,
            kernel_size=[11],
            stride=[1],
            dilation=[1],
            dropout=0.0,
            residual=False,
            groups=1,
            separable=True,
            heads=-1,
            residual_mode=residual_mode,
            normalization=normalization_mode,
            norm_groups=norm_groups,
            activation=activation,
            residual_panes=[],
            conv_mask=conv_mask,
            kernel_size_factor=1.0,
            stride_last=False
        )
        encoder_layers.append(self.conv_block_1)
        feat_in = 128

                
        self.conv_res_block_1 = ConvBlock(
            feat_in,
            64,
            repeat=2,
            kernel_size=[13],
            stride=[1],
            dilation=[1],
            dropout=0.0,
            residual=True,
            groups=1,
            separable=True,
            heads=-1,
            residual_mode=residual_mode,
            normalization=normalization_mode,
            norm_groups=norm_groups,
            activation=activation,
            residual_panes=[],
            conv_mask=conv_mask,
            kernel_size_factor=1.0,
            stride_last=False
        )
        encoder_layers.append(self.conv_res_block_1)
        feat_in = 64


        self.conv_res_block_2 = ConvBlock(
            feat_in,
            64,
            repeat=2,
            kernel_size=[15],
            stride=[1],
            dilation=[1],
            dropout=0.0,
            residual=True,
            groups=1,
            separable=True,
            heads=-1,
            residual_mode=residual_mode,
            normalization=normalization_mode,
            norm_groups=norm_groups,
            activation=activation,
            residual_panes=[],
            conv_mask=conv_mask,
            kernel_size_factor=1.0,
            stride_last=False
        )
        encoder_layers.append(self.conv_res_block_2)
        feat_in = 64

        self.conv_res_block_3 = ConvBlock(
            feat_in,
            64,
            repeat=2,
            kernel_size=[17],
            stride=[1],
            dilation=[1],
            dropout=0.0,
            residual=True,
            groups=1,
            separable=True,
            heads=-1,
            residual_mode=residual_mode,
            normalization=normalization_mode,
            norm_groups=norm_groups,
            activation=activation,
            residual_panes=[],
            conv_mask=conv_mask,
            kernel_size_factor=1.0,
            stride_last=False
        )
        encoder_layers.append(self.conv_res_block_3)
        feat_in = 64

        
        # self.conv_res_block_4 = ConvBlock(
        #     feat_in,
        #     64,
        #     repeat=2,
        #     kernel_size=[19],
        #     stride=[1],
        #     dilation=[1],
        #     dropout=0.0,
        #     residual=True,
        #     groups=1,
        #     separable=True,
        #     heads=-1,
        #     residual_mode=residual_mode,
        #     normalization=normalization_mode,
        #     norm_groups=norm_groups,
        #     activation=activation,
        #     residual_panes=[],
        #     conv_mask=conv_mask,
        #     kernel_size_factor=1.0,
        #     stride_last=False
        # )
        # encoder_layers.append(self.conv_res_block_4)
        # feat_in = 64


        # self.conv_res_block_5 = ConvBlock(
        #     feat_in,
        #     64,
        #     repeat=2,
        #     kernel_size=[21],
        #     stride=[1],
        #     dilation=[1],
        #     dropout=0.0,
        #     residual=True,
        #     groups=1,
        #     separable=True,
        #     heads=-1,
        #     residual_mode=residual_mode,
        #     normalization=normalization_mode,
        #     norm_groups=norm_groups,
        #     activation=activation,
        #     residual_panes=[],
        #     conv_mask=conv_mask,
        #     kernel_size_factor=1.0,
        #     stride_last=False
        # )
        # encoder_layers.append(self.conv_res_block_5)
        # feat_in = 64

        # self.conv_res_block_6 = ConvBlock(
        #     feat_in,
        #     64,
        #     repeat=2,
        #     kernel_size=[23],
        #     stride=[1],
        #     dilation=[1],
        #     dropout=0.0,
        #     residual=True,
        #     groups=1,
        #     separable=True,
        #     heads=-1,
        #     residual_mode=residual_mode,
        #     normalization=normalization_mode,
        #     norm_groups=norm_groups,
        #     activation=activation,
        #     residual_panes=[],
        #     conv_mask=conv_mask,
        #     kernel_size_factor=1.0,
        #     stride_last=False
        # )
        # encoder_layers.append(self.conv_res_block_6)
        # feat_in = 64
        

        self.conv_block_2 = ConvBlock(
            feat_in,
            128,
            repeat=1,
            kernel_size=[29],
            stride=[1],
            dilation=[2],
            dropout=0.0,
            residual=False,
            groups=1,
            separable=True,
            heads=-1,
            residual_mode=residual_mode,
            normalization=normalization_mode,
            norm_groups=norm_groups,
            activation=activation,
            residual_panes=[],
            conv_mask=conv_mask,
            kernel_size_factor=1.0,
            stride_last=False
        )
        encoder_layers.append(self.conv_block_2)
        feat_in = 128

        self.conv_block_3 = ConvBlock(
            feat_in,
            128,
            repeat=1,
            kernel_size=[1],
            stride=[1],
            dilation=[1],
            dropout=0.0,
            residual=False,
            groups=1,
            separable=True,
            heads=-1,
            residual_mode=residual_mode,
            normalization=normalization_mode,
            norm_groups=norm_groups,
            activation=activation,
            residual_panes=[],
            conv_mask=conv_mask,
            kernel_size_factor=1.0,
            stride_last=False
        )
        encoder_layers.append(self.conv_block_3)
        feat_in = 128

        self._feat_out = feat_in

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal, length=None):
        s_input, length = self.encoder(([audio_signal.squeeze()], length))
        if length is None:
            return s_input[-1]

        return s_input[-1], length
