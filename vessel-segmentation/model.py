from typing import Optional, OrderedDict

from torch.nn import (
    Module,
    ReLU,
    _Conv3d,
    Sequential,
    Dropout,
    MaxPool2d,
    init,
    Upsample,
    Softmax,
)

from ..params import ModelParams


class Conv3d(_Conv3d):
    r"""Convolutional layer with custom initialization of learnable parameters."""

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.constant_(self.bias, 0.1)


def conv_layer(
    _in_channels: int,
    _out_channels: int,
    kernel_size=3,
    stride=1,
    dilation=1,
    padding=None,
):
    modules = OrderedDict()
    modules["conv1"] = Conv3d(
        in_channels=_in_channels,
        out_channels=_out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )
    modules["activation1"] = ReLU(inplace=True)
    modules["dropout"] = Dropout(p=0.2)

    modules["conv2"] = Conv3d(
        in_channels=_in_channels,
        out_channels=_out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )
    modules["activation2"] = ReLU(inplace=True)
    return Sequential(modules)


class RetinaUNetModel(Module):
    """PyTorch implementation of RetinaUNet model.
    Reference: https://github.com/orobix/retina-unet

    """

    def __init__(self, in_channels, params: Optional[ModelParams] = None):
        super().__init__()

        if params is None:
            params = ModelParams()

        m = params.feature_map_factor

        # Encoding layers
        self.layer_1 = conv_layer(1 * in_channels, 1 * m)
        self.layer_2 = conv_layer(1 * m, 2 * m)
        self.layer_3 = conv_layer(2 * m, 4 * m)

        # Decoding layers
        self.layer_4_1 = Upsample(scale_factor=2)
        self.layer_4_2 = conv_layer(4 * m, 2 * m)

        self.layer_5_1 = Upsample(scale_factor=2)
        self.layer_5_2 = conv_layer(2 * m, 1 * m)

        last_module = OrderedDict()
        last_module["conv"] = Conv3d(in_channels=1 * m, out_channels=2, kernel_size=1)
        last_module["activation"] = ReLU(inplace=True)
        self.layer_6 = Sequential(last_module)

    def forward(self, x):
        x = x_1 = self.layer_1(x)
        x = MaxPool2d(2)(x)
        x = x_2 = self.layer_2(x)
        x = MaxPool2d(2)(x)

        x = self.layer_3(x)

        x = self.layer_4_1(x)
        x = self.layer_4_2(x_2 + x)

        x = self.layer_5_1(x)
        x = self.layer_5_2(x_1 + x)

        x = self.layer_6(x)

        return Softmax()(x)
