from typing import Union, List, Dict, Any

import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.utils.cpp_extension import load
import types
import common
cutlass = load(name="cutlass_sparse", sources=["./models/models_sparse/gemm.cu"], extra_include_paths=["./models/models_sparse/"], build_directory="./models/models_sparse/")
zero_bias = torch.zeros((512)).cuda()

def free():
    cutlass.free()

def get_cutlass():
    return cutlass

def conv_forward(self, input: torch.Tensor):
    return cutlass.conv_forward(input, self.weight, self.bias, self.weight_c, self.flags, self.stride[0], self.padding[0])

def linear_forward(self, input: torch.Tensor) -> torch.Tensor:
    return cutlass.linear_forward(input, self.weight, self.bias, self.flags)

def construct(net:nn.modules, tile_shape: List=[128,1]):
    layers = common.get_layers(net)
    for layer in layers:
        if type(layer) is nn.Conv2d:
            weight = torch.clone(layer.weight)
            if type(layer.bias) is None:
                bias = torch.clone(layer.bias)
            print("Starting generate flags for {}...".format(layer))
            flags = cutlass.gen_flag(weight.view((weight.shape[0], -1)), tile_shape[1], tile_shape[0])
            print(flags)
            print("Starting change data format from row-major to column major...")
            weight_c = cutlass.r2c(weight.view((weight.shape[0], -1)))
            layer.register_buffer("flags", flags)
            layer.register_buffer("weight_c", weight_c)
            with torch.no_grad():
                layer.weight = nn.Parameter(weight)
                if type(layer.bias) is None:
                    layer.bias = nn.Parameter(bias)
            layer.forward = types.MethodType(conv_forward, layer)
        elif type(layer) is nn.Linear:
            weight = torch.clone(layer.weight)
            bias = torch.clone(layer.bias)
            print("Starting generate flags for {}...".format(layer))
            flags = cutlass.gen_flag(weight.view((weight.shape[0], -1)), tile_shape[1], tile_shape[0])
            print(flags)
            layer.register_buffer("flags", flags)
            with torch.no_grad():
                layer.weight = nn.Parameter(weight)
                layer.bias = nn.Parameter(bias)
            layer.forward = types.MethodType(linear_forward, layer)


class Conv_flags(torch.nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv_flags, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.d_output = None

    def forward(self, input: torch.Tensor):
        return cutlass.conv_forward(input, self.weight, self.bias, self.weight_c, self.flags, self.stride[0], self.padding[0])

    def construct(self, conv: nn.Conv2d ,tile_height, tile_width):
        weight = torch.clone(conv.weight)
        bias = torch.clone(conv.bias)
        print("Starting generate flags for {}...".format(self))
        flags = cutlass.gen_flag(weight.view((weight.shape[0], -1)), tile_width, tile_height)
        print(flags)
        print("Starting change data format from row-major to column major...")
        weight_c = cutlass.r2c(weight.view((weight.shape[0], -1)))
        self.register_buffer("flags", flags)
        self.register_buffer("weight_c", weight_c)
        with torch.no_grad():
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias)

    def add_flags(self, tile_height, tile_width):
        weight = self.weight.view((self.weight.shape[0], -1))
        flags_shape = [(weight.shape[0]+tile_height-1)//tile_height, (weight.shape[1]+tile_width-1)//tile_width + 1]
        self.register_buffer("flags", torch.zeros(flags_shape, dtype=torch.int32))
        self.register_buffer("weight_c", torch.zeros((weight.shape[1], weight.shape[0])))


class Linear_flags(torch.nn.modules.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(torch.nn.modules.Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cutlass.linear_forward(input, self.weight, self.bias, self.flags)

    def construct(self, linear: nn.Linear, tile_height, tile_width):
        weight = torch.clone(linear.weight)
        bias = torch.clone(linear.bias)
        print("Starting generate flags for {}...".format(self))
        flags = cutlass.gen_flag(weight.view((weight.shape[0], -1)), tile_width, tile_height)
        print(flags)
        print("Starting change data format from row-major to column major...")
        self.register_buffer("flags", flags)
        with torch.no_grad():
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(bias)

    def add_flags(self, tile_height, tile_width):
        weight = self.weight.view((self.weight.shape[0], -1))
        flags_shape = [(weight.shape[0]+tile_height-1)//tile_height, (weight.shape[1]+tile_width-1)//tile_width + 1]
        self.register_buffer("flags", torch.zeros(flags_shape, dtype=torch.int32))


class AlexNet_flags(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            Conv_flags(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_flags(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_flags(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv_flags(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv_flags(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear_flags(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Linear_flags(4096, 4096),
            nn.ReLU(inplace=True),
            Linear_flags(4096, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x