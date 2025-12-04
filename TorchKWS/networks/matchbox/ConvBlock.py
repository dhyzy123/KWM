from typing import List, Optional, Tuple

from torch import nn
from networks.matchbox.MaskedConv1d import MaskedConv1d
from torch.nn.init import _calculate_correct_fan
import torch
import math

conv_activations = {"hardtanh": nn.Hardtanh, "relu": nn.ReLU, "selu": nn.SELU, "silu": nn.SiLU}

def tds_uniform_(tensor, mode='fan_in'):
    """
    Uniform Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -
    .. math::
        \text{bound} = \text{2} \times \sqrt{\frac{1}{\text{fan\_mode}}}
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0  # sqrt(4.0) = 2
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def tds_normal_(tensor, mode='fan_in'):
    """
    Normal Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -
    .. math::
        \text{bound} = \text{2} \times \sqrt{\frac{1}{\text{fan\_mode}}}
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.normal_(0.0, bound)

def init_weights(m, mode: Optional[str] = 'xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode is not None:
            if mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif mode == 'tds_uniform':
                tds_uniform_(m.weight)
            elif mode == 'tds_normal':
                tds_normal_(m.weight)
            else:
                raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def compute_new_kernel_size(kernel_size, kernel_width):
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def get_same_padding(kernel_size, stride, dilation) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2

class GroupShuffle(nn.Module):
    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x

class ConvBlock(nn.Module):
    """
    Args:
        inplanes: Number of input channels.
        planes: Number of output channels.
        repeat: Number of repeated sub-blocks (R) for this block.
        kernel_size: Convolution kernel size across all repeated sub-blocks.
        kernel_size_factor: Floating point scale value that is multiplied with kernel size,
            then rounded down to nearest odd integer to compose the kernel size. Defaults to 1.0.
        stride: Stride of the convolutional layers.
        dilation: Integer which defined dilation factor of kernel. Note that when dilation > 1, stride must
            be equal to 1.
        padding: String representing type of padding. Currently only supports "same" padding,
            which symmetrically pads the input tensor with zeros.
        dropout: Floating point value, determins percentage of output that is zeroed out.
        activation: String representing activation functions. Valid activation functions are :
            {"hardtanh": nn.Hardtanh, "relu": nn.ReLU, "selu": nn.SELU, "swish": Swish}.
            Defaults to "relu".
        residual: Bool that determined whether a residual branch should be added or not.
            All residual branches are constructed using a pointwise convolution kernel, that may or may not
            perform strided convolution depending on the parameter `residual_mode`.
        groups: Number of groups for Grouped Convolutions. Defaults to 1.
        separable: Bool flag that describes whether Time-Channel depthwise separable convolution should be
            constructed, or ordinary convolution should be constructed.
        heads: Number of "heads" for the masked convolution. Defaults to -1, which disables it.
        normalization: String that represents type of normalization performed. Can be one of
            "batch", "group", "instance" or "layer" to compute BatchNorm1D, GroupNorm1D, InstanceNorm or
            LayerNorm (which are special cases of GroupNorm1D).
        norm_groups: Number of groups used for GroupNorm (if `normalization` == "group").
        residual_mode: String argument which describes whether the residual branch should be simply
            added ("add") or should first stride, then add ("stride_add"). Required when performing stride on
            parallel branch as well as utilizing residual add.
        residual_panes: Number of residual panes, used for Jasper-DR models. Please refer to the paper.
        conv_mask: Bool flag which determines whether to utilize masked convolutions or not. In general,
            it should be set to True.
        stride_last: Bool flag that determines whether all repeated blocks should stride at once,
            (stride of S^R when this flag is False) or just the last repeated block should stride
            (stride of S when this flag is True).
    """

    __constants__ = ["conv_mask", "separable", "residual_mode", "res", "mconv"]

    def __init__(
        self,
        inplanes,
        planes,
        repeat=3,
        kernel_size=11,
        kernel_size_factor=1,
        stride=1,
        dilation=1,
        padding='same',
        dropout=0.2,
        activation=None,
        residual=True,
        groups=1,
        separable=False,
        heads=-1,
        normalization="batch",
        norm_groups=1,
        residual_mode='add',
        residual_panes=[],
        conv_mask=False,
        stride_last=False
    ):
        super(ConvBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        kernel_size_factor = float(kernel_size_factor)
        if type(kernel_size) in (list, tuple):
            kernel_size = [compute_new_kernel_size(k, kernel_size_factor) for k in kernel_size]
        else:
            kernel_size = compute_new_kernel_size(kernel_size, kernel_size_factor)

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])

        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode

        inplanes_loop = inplanes
        conv = nn.ModuleList()

        for _ in range(repeat - 1):
            # Stride last means only the last convolution in block will have stride
            if stride_last:
                stride_val = [1]
            else:
                stride_val = stride

            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride_val,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    heads=heads,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups
                )
            )

            conv.extend(self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

            inplanes_loop = planes

        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups
            )
        )

        self.mconv = conv

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()

            if residual_mode == 'stride_add':
                stride_val = stride
            else:
                stride_val = [1]

            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res = nn.ModuleList(
                    self._get_conv_bn_layer(
                        ip,
                        planes,
                        kernel_size=1,
                        normalization=normalization,
                        norm_groups=norm_groups,
                        stride=stride_val
                    )
                )

                res_list.append(res)

            self.res = res_list
        else:
            self.res = None

        self.mout = nn.Sequential(*self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

    def _get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
    ):
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
                heads=heads,
                use_mask=use_mask
            )
        else:
            return nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
            )
                

    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        normalization="batch",
        norm_groups=1
    ):
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                self._get_conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=in_channels,
                    heads=heads
                ),
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=bias,
                    groups=groups
                ),
            ]
        else:
            layers = [
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups
                )
            ]

        if normalization == "group":
            layers.append(nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels))
        elif normalization == "instance":
            layers.append(nn.GroupNorm(num_groups=out_channels, num_channels=out_channels))
        elif normalization == "layer":
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match" f" one of [batch, layer, group, instance]."
            )

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [activation, nn.Dropout(p=drop_prob)]
        return layers

    def forward(self, input_: Tuple[List[torch.Tensor], Optional[torch.Tensor]]):
        """
        Forward pass of the module.
        Args:
            input_: The input is a tuple of two values - the preprocessed audio signal as well as the lengths
                of the audio signal. The audio signal is padded to the shape [B, D, T] and the lengths are
                a torch vector of length B.
        Returns:
            The output of the block after processing the input through `repeat` number of sub-blocks,
            as well as the lengths of the encoded audio after padding/striding.
        """
        # type: (Tuple[List[Tensor], Optional[Tensor]]) -> Tuple[List[Tensor], Optional[Tensor]] # nopep8
        lens_orig = None
        xs = input_[0]
        if len(input_) == 2:
            xs, lens_orig = input_

        # compute forward convolutions
        out = xs[-1]

        lens = lens_orig
        for i, l in enumerate(self.mconv):
            # if we're doing masked convolutions, we need to pass in and
            # possibly update the sequence lengths
            # if (i % 4) == 0 and self.conv_mask:
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)

                if self.residual_mode == 'add' or self.residual_mode == 'stride_add':
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)

        # compute the output
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens

        return [out], lens