import os
from typing import List, Any

import torch
import torch.nn as nn
from common import get_layers

class Metrics():
    pass

class MetricsL1(Metrics):
    @staticmethod
    def eval(layer: nn.Module, *agrs: Any, **kargs: Any):
        return torch.abs(layer.weight)
    
    @staticmethod
    def hook_func(*args: Any, **kargs: Any):
        pass

class MetricsL2(Metrics):
    @staticmethod
    def eval(layer: nn.Module, *agrs: Any, **kargs: Any):
        return torch.square(layer.weight)
    
    @staticmethod
    def hook_func(*args: Any, **kargs: Any):
        pass


class MetricsGradient(Metrics):
    @staticmethod
    def eval(layer: nn.Module, grad: torch.Tensor=None, *args: Any, **kargs: Any):
        return torch.square(layer.weight.cpu()*grad.cpu())

    @staticmethod
    def hook_func(net: nn.Module, *args: Any, **kargs: Any):
        with torch.no_grad():
            layers = get_layers(net)
            grads = kargs["grads"]
            total_steps = kargs["total_steps"]
            worker_nums = kargs["worker_nums"]
            step = kargs["step"]
            for i in range(len(grads)):
                grads[i] = grads[i].cuda()
            if step == 0:
                for i in range(len(grads)):
                    grads[i][:] = 0
            for l,g in zip(layers, grads):
                g += l.weight.grad/total_steps

    @staticmethod
    def create_grads(net: nn.Module) -> List[torch.Tensor]: 
        layers = get_layers(net)
        grads = []
        for l in layers:
            grad = torch.zeros_like(l.weight)
            grads.append(grad)
        return grads
        