import os
from typing import List, Any
from itertools import zip_longest

import torch
from torch.nn import modules
from torch.utils.data.dataloader import DataLoader
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.models as models
import torch.distributed as dist
import common

def imply_mask(self, input):
    with torch.no_grad():
        self.weight = nn.Parameter(self.weight*self.mask)

def add_masks(net):
    layers = common.get_layers(net)
    with torch.no_grad():
        masks = [torch.ones(layer.weight.shape) for layer in layers]
        for i in range(len(masks)):
            layers[i].register_buffer('mask', masks[i])
            layers[i].register_forward_pre_hook(imply_mask)

class Method():
    def __init__(self, metrics=None) -> None:
        self._metrics = metrics     # Class for calculating the importance score of weight matrix.
    
    def __call__(self, net:nn.Module, sparsity:float, is_soft:bool=False) -> None:
        pass
    
class Pruner():
    def __init__(self, method:Method=None) -> None:
        self._method = method       # Class to prune the weight matrix.

    def prune(self, net:torch.nn.Module, sparsity:float, grads: List[torch.Tensor] = [], is_soft: bool=False) -> None:
        if not self._method:
            raise Exception("Methos is not set! ")
        if sparsity > 1 or sparsity <0:
            raise ValueError("Sparsity must be a float number between 0 and 1.")
        self._method(net, sparsity, grads)

        
class EW_pruning(Method):
    def __init__(self, metrics=None) -> None:
        super().__init__(metrics=metrics)

    def __call__(self, net:nn.Module, sparsity: float, grads:List[torch.Tensor]=[], is_soft:bool=False) -> None:
        layers = common.get_layers(net)
        with torch.no_grad():
            to_cat = [self._metrics.eval(l, grad=g).view(-1) for l,g in zip_longest(layers, grads)]
            concatenated_weights = torch.cat(to_cat)
            target_index = int(sparsity*len(concatenated_weights))
            
            # del to_cat
            # torch.cuda.empty_cache()

            sorted_result, _ = torch.sort(concatenated_weights)
            threshold = sorted_result[target_index].item()

            # del concatenated_weights
            # del sorted_result
            # torch.cuda.empty_cache()

            to_cat = [self._metrics.eval(l, grad=g).view(-1) for l,g in zip_longest(layers, grads)]
            masks = [c.view(l.weight.shape) >= threshold for l,c in zip_longest(layers, to_cat)]
            
            # del to_cat
            # torch.cuda.empty_cache()
            

        for l,m in zip(layers, masks):
            l.register_buffer("mask", m)
            l.register_forward_pre_hook(imply_mask)


class TW_pruning(Method):
    def __init__(self, tile_shape:list=[128,1], metrics=None) -> None:
        super().__init__(metrics=metrics)
        self.tile_shape = tile_shape
    
    def __call__(self, net: nn.Module, sparsity: float, grads:List[torch.Tensor]=[], is_soft:bool=False) -> None:
        def cal_tile_mean(weight, tile_shape):
            tile_mean = torch.zeros((int((weight.shape[0] + tile_shape[0] - 1)/tile_shape[0]), int((weight.shape[1] + tile_shape[1] - 1)/tile_shape[1])))
            for i in range(tile_mean.shape[0]):
                for j in range(tile_mean.shape[1]):
                    tile = torch.abs(weight[i*tile_shape[0]:(i+1)*tile_shape[0], j*tile_shape[1]:(j+1)*tile_shape[1]])
                    tile_mean[i][j] = torch.sum(tile) / float(tile.nelement())
            return tile_mean

        layers = common.get_layers(net)
        tile_shape = self.tile_shape
        with torch.no_grad():
            to_cat = []
            shape_list = []
            weights = [self._metrics.eval(l, grad=g) for l,g in zip_longest(layers, grads)]
            for w,layer in zip(weights, layers):
                if type(layer) is nn.Conv2d:
                    tile_mean = cal_tile_mean(w.view((layer.weight.shape[0],-1)), tile_shape)
                elif type(layer) is nn.Linear:
                    tile_mean = cal_tile_mean(w, tile_shape)
                else:
                    tile_mean = cal_tile_mean(w, tile_shape)
                shape_list.append(tile_mean.shape)
                to_cat.append(tile_mean.view(-1))

            if len(to_cat)==1:
                concatenated_weights = to_cat[0]
            else:
                concatenated_weights = torch.cat(to_cat)

            target_index = int(sparsity*len(concatenated_weights))
            sorted_result, _ = torch.sort(concatenated_weights)
            threshold = sorted_result[target_index]

            tile_masks = [w.view(s) >= threshold for w,s in zip(to_cat, shape_list)]
            
            for m,l in zip(tile_masks, layers):
                if type(l) is nn.Conv2d:
                    mask = torch.zeros_like(l.weight).view((l.weight.shape[0],-1))
                elif type(l) is nn.Linear:
                    mask = torch.zeros_like(l.weight)
                else:
                    mask = torch.zeros_like(l.weight)
                for ii in range(m.shape[0]):
                    for jj in range(m.shape[1]):
                        if m[ii][jj]:
                            mask[ii*tile_shape[0]:(ii+1)*tile_shape[0], jj*tile_shape[1]:(jj+1)*tile_shape[1]] = 1
                if type(l) is nn.Conv2d:
                    mask = mask.view(l.weight.shape)
                l.register_buffer("mask", mask)
                l.register_forward_pre_hook(imply_mask)


class TEW_pruning(Method):
    def __init__(self, tile_shape:list=[128,1], metrics=None) -> None:
        super().__init__(metrics=metrics)
        self.tile_shape = tile_shape
    
    def __call__(self, net: nn.Module, sparsity: float, recovery:float=0.01, is_soft:bool=False) -> None:
        def cal_tile_mean(weight, tile_shape):
            tile_mean = torch.zeros((int((weight.shape[0] + tile_shape[0] - 1)/tile_shape[0]), int((weight.shape[1] + tile_shape[1] - 1)/tile_shape[1])))
            for i in range(tile_mean.shape[0]):
                for j in range(tile_mean.shape[1]):
                    tile = torch.abs(weight[i*tile_shape[0]:(i+1)*tile_shape[0], j*tile_shape[1]:(j+1)*tile_shape[1]])
                    tile_mean[i][j] = torch.sum(tile) / float(tile.nelement())
            return tile_mean

        layers = common.get_layers(net)
        tile_shape = self.tile_shape
        with torch.no_grad():
            to_cat = []
            shape_list = []
            weights = [self._metrics(l) for l in layers]
            for w,layer in zip(weights, layers):
                if type(layer) is nn.Conv2d:
                    tile_mean = cal_tile_mean(w.view((layer.weight.shape[0],-1)), tile_shape)
                elif type(layer) is nn.Linear:
                    tile_mean = cal_tile_mean(w, tile_shape)
                else:
                    tile_mean = cal_tile_mean(w, tile_shape)
                shape_list.append(tile_mean.shape)
                to_cat.append(tile_mean.view(-1))

            if len(to_cat)==1:
                concatenated_weights = to_cat[0]
            else:
                concatenated_weights = torch.cat(to_cat)

            if sparsity+recovery>1:
                raise ValueError("Sparsity({}) plus recovery({}) is larger than 1.".format(sparsity, recovery))
            target_index = int((sparsity+recovery)*len(concatenated_weights))
            sorted_result, _ = torch.sort(concatenated_weights)
            threshold = sorted_result[target_index]

            tile_masks = [w.view(s) >= threshold for w,s in zip(to_cat, shape_list)]

            to_cat = []
            recovery_masks = []
            for m,l in zip(tile_masks, layers):
                if type(l) is nn.Conv2d:
                    mask = torch.zeros_like(l.weight).view((l.weight.shape[0],-1))
                elif type(l) is nn.Linear:
                    mask = torch.zeros_like(l.weight)
                else:
                    mask = torch.zeros_like(l.weight)
                for ii in range(m.shape[0]):
                    for jj in range(m.shape[1]):
                        if m[ii][jj]:
                            mask[ii*tile_shape[0]:(ii+1)*tile_shape[0], jj*tile_shape[1]:(jj+1)*tile_shape[1]] = 1
                if type(l) is nn.Conv2d:
                    mask = mask.view(l.weight.shape)
                recovery_mask = torch.logical_xor(mask, torch.ones_like(mask))
                to_cat.append((self._metrics(l)*recovery_mask).flatten())
                recovery_masks.append(recovery_mask)
                l.register_buffer("mask", mask)
                l.register_forward_pre_hook(imply_mask)
            
            # Recovery step

            if len(to_cat)==1:
                concatenated_recovery_weights = to_cat[0]
            else:
                concatenated_recovery_weights = torch.cat(to_cat)
            target_index = int((1-recovery)*len(concatenated_recovery_weights))
            sorted_result,_ = torch.sort(concatenated_recovery_weights)
            threshold = sorted_result[target_index]

            for m,l in zip(recovery_masks, layers):
                recovery_mask = (l.weight*m)>=threshold
                l.mask += recovery_mask