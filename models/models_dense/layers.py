import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.utils.cpp_extension import load
cutlass = load(name="cutlass_dense", sources=["./models/models_dense/gemm.cu"], extra_include_paths=["./models/models_dense/"], build_directory="./models/models_dense/")

def free():
    cutlass.free()

class Conv_cutlass(torch.nn.modules.conv._ConvNd):
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
        super(Conv_cutlass, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.d_output = None

    def forward(self, input: torch.Tensor):
        return cutlass.conv_forward(input, self.weight, self.bias, self.stride[0], self.padding[0])

class AlexNet_cutlass(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            Conv_cutlass(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_cutlass(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_cutlass(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv_cutlass(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv_cutlass(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x