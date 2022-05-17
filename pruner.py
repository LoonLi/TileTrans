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
from dst import get_layers
import torch.distributed as dist

from models.models_torch.layers import alexnet
from metrics import Metrics, MetricsGradient, MetricsL1
from reconstructor import Reconstructor, ReconMethod, ReconMethodL1Sort
import common
import train
import dst



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

def check_sparsity(layer):
    print("Sum = {}, nele = {}, Sparsity = {}".format(torch.sum(layer.mask),layer.mask.nelement(), 1 - torch.sum(layer.mask)/float(layer.mask.nelement())))
    return (1 - torch.sum(layer.mask)/float(layer.mask.nelement())).item()

def check_model_saprsity(net):
    result = []
    layers = get_layers(net)
    for l in layers:
        print(l)
        result.append(check_sparsity(l))
    return result



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


def steps_prune(net:torch.nn.Module, sparsity_per_step: float = 0.02, train_loader: Data.DataLoader = None, test_loader: DataLoader = None):
    SPARSITY = 0.9        
    SPARSITY_PER_STEP = sparsity_per_step
    
    metrics = MetricsL1()
    method = TW_pruning([128,1] ,metrics)
    pruner = Pruner(method)

    total_sparsity = 0
    while total_sparsity < SPARSITY:
        total_sparsity = min(total_sparsity+SPARSITY_PER_STEP, SPARSITY)
        print("Start pruning model to sparity {}...".format(total_sparsity))
        pruner.prune(net, total_sparsity)
        # print("Start retaining...")
        # pruner.retrain(net)


def steps_prune_dst(net:torch.nn.Module, 
                    sparsity:int=0.5,
                    start_sparsity:float=0, 
                    metrics: Metrics = MetricsGradient, 
                    method: Method = TW_pruning, 
                    tile_shape: List = [128, 1] , 
                    sparsity_per_step: float = 0.02,
                    per_worker_batch_size=32,
                    lr:float=0.01,  
                    fine_tune_epochs: int = 1,
                    early_stop: float=None,
                    steps_stop: int = None,
                    is_reconstruct: bool=False, 
                    reconstruct_method: ReconMethod=ReconMethodL1Sort, 
                    worker_nums: int = 2, 
                    master:str="c2",
                    log_server_port:int=19999):

    if master:
        rank = int(os.environ["SLURM_PROCID"])
    dst.set_gpu(rank)

    SPARSITY = sparsity        
    SPARSITY_PER_STEP = sparsity_per_step
    WOKER_NUMS = worker_nums
    MASTER = master

    metrics = metrics
    if method in [TW_pruning, TEW_pruning]:
        method = method(tile_shape ,metrics)
    else:
        method = EW_pruning(metrics)
    pruner = Pruner(method)

    if metrics is MetricsGradient:
        grads = metrics.create_grads(net)
        print("Start warming up...")
        dst.distributed_train(net, 
                            epochs=1,
                            steps_stop=steps_stop, 
                            per_worker_batch_size=per_worker_batch_size, 
                            lr=0.0001,
                            worker_nums=WOKER_NUMS, 
                            master=MASTER, 
                            find_unused_parameters=True,
                            log_server_port=log_server_port,
                            after_backward_hook=metrics.hook_func, 
                            grads=grads)
    else:
        grads = []


    for g in grads:
        dist.all_reduce(g)
        g = g/worker_nums



    total_sparsity = start_sparsity
    while total_sparsity < SPARSITY:

        if is_reconstruct:
            print("Start reconstructing...")
            recon = Reconstructor(metrics=metrics, method=reconstruct_method)
            net.cpu()
            recon(net, grads)
            net.cuda()
            
        total_sparsity = min(total_sparsity+SPARSITY_PER_STEP, SPARSITY)
        print("Start pruning model to sparity {}...".format(total_sparsity))
        net.cpu()
        pruner.prune(net, total_sparsity, grads=grads)
        net.cuda()
        print("Finished pruning the sparsity to {}...".format(total_sparsity))
        # dist.barrier()
        if metrics is MetricsGradient:
            grads = metrics.create_grads(net)
        else:
            grads = []
        print("Start retaining...")
        dst.distributed_train(net, save_name="work_space/{model}_{metrics}_{pruing_method}_ReconstuctIs{is_reconstruct}_{reconstruct_method}_{sparsity}".format(model=type(net).__name__, metrics=metrics.__name__, pruing_method=type(method).__name__, is_reconstruct=is_reconstruct, reconstruct_method=reconstruct_method.__name__, sparsity=int(total_sparsity*100)), 
                        epochs=fine_tune_epochs,
                        steps_stop=steps_stop,
                        early_stop=early_stop, 
                        per_worker_batch_size=per_worker_batch_size, 
                        lr=lr, 
                        worker_nums=WOKER_NUMS, 
                        master=MASTER,
                        log_server_port=log_server_port, 
                        find_unused_parameters=True, 
                        after_backward_hook=metrics.hook_func, 
                        grads=grads)
        for g in grads:
            dist.all_reduce(g)
            g = g/worker_nums
        common.check_model_saprsity(net)
        


def one_shot_pruning(net:torch.nn.Module, 
                    sparsity:int=0.5,
                    metrics: Metrics = MetricsL1, 
                    method: Method = TW_pruning, 
                    tile_shape: List = [128, 1] , 
                    per_worker_batch_size=32,
                    lr:float=0.01,  
                    fine_tune_epochs: int = 1,
                    early_stop: float=None,
                    steps_stop: int = None,
                    is_reconstruct: bool=False, 
                    reconstruct_method: ReconMethod=ReconMethodL1Sort, 
                    worker_nums: int = 2, 
                    master:str="c2",
                    log_server_port:int=19999):

    SPARSITY = sparsity        
    WOKER_NUMS = worker_nums
    MASTER = master

    metrics = metrics
    if method in [TW_pruning, TEW_pruning]:
        method = method(tile_shape ,metrics)
    else:
        method = EW_pruning(metrics)
    pruner = Pruner(method)

    if metrics is MetricsGradient:
        grads = metrics.create_grads(net)
        print("Start warming up...")
        dst.distributed_train(net, 
                            epochs=1,
                            steps_stop=steps_stop, 
                            per_worker_batch_size=per_worker_batch_size, 
                            lr=0.0001,
                            worker_nums=WOKER_NUMS, 
                            master=MASTER, 
                            find_unused_parameters=True,
                            log_server_port=log_server_port,
                            after_backward_hook=metrics.hook_func, 
                            grads=grads)
    else:
        grads = []

    for g in grads:
        dist.all_reduce(g)
        g = g/worker_nums

    if is_reconstruct:
        print("Start reconstructing...")
        recon = Reconstructor(metrics=metrics, method=reconstruct_method)
        net.cpu()
        recon(net, grads)
        net.cuda()

    print("Start pruning model to sparity {}...".format(SPARSITY))
    net.cpu()
    pruner.prune(net, SPARSITY, grads=grads)
    net.cuda()
    print("Finished pruning the sparsity to {}...".format(SPARSITY))
    # dist.barrier()
    if metrics is MetricsGradient:
        grads = metrics.create_grads(net)
    else:
        grads = []
    print("Start retaining...")
    dst.distributed_train(net, save_name="work_space/{model}_{metrics}_{pruing_method}_ReconstuctIs{is_reconstruct}_{reconstruct_method}_{sparsity}".format(model=type(net).__name__, metrics=metrics.__name__, pruing_method=type(method).__name__, is_reconstruct=is_reconstruct, reconstruct_method=reconstruct_method.__name__, sparsity=int(SPARSITY*100)), 
                    epochs=fine_tune_epochs,
                    steps_stop=steps_stop,
                    early_stop=early_stop, 
                    per_worker_batch_size=per_worker_batch_size, 
                    lr=lr, 
                    worker_nums=WOKER_NUMS, 
                    master=MASTER,
                    log_server_port=log_server_port, 
                    find_unused_parameters=True, 
                    after_backward_hook=metrics.hook_func, 
                    grads=grads)
    common.check_model_saprsity(net)


if __name__ == "__main__":

    # net = models.vgg16(pretrained=True)
    # add_masks(net)
    # backup = net.state_dict()

    # for i in range(1,10):
    #     sparsity = float(i)/10
    #     one_shot_pruning(net, 
    #                     sparsity=sparsity,
    #                     metrics = MetricsL1, 
    #                     method = EW_pruning, 
    #                     tile_shape = [128, 1] , 
    #                     per_worker_batch_size=16,
    #                     lr=0.01,  
    #                     fine_tune_epochs = 1,
    #                     early_stop = None,
    #                     steps_stop = 1,
    #                     is_reconstruct = False, 
    #                     worker_nums = 8, 
    #                     master = "c5",
    #                     log_server_port = 19999)
    #     net.load_state_dict(backup)

    net = models.alexnet(pretrained=True)
    steps_prune_dst(net, 
                sparsity=0.9,
                start_sparsity=0, 
                metrics= MetricsL1, 
                method = TW_pruning, 
                tile_shape = [32, 32], 
                sparsity_per_step = 0.05,
                per_worker_batch_size=64,
                lr=0.01,  
                fine_tune_epochs = 10,
                early_stop= 0.01,
                steps_stop= None,
                is_reconstruct= True, 
                reconstruct_method=ReconMethodL1Sort, 
                worker_nums = 8, 
                master ="c7",
                log_server_port=19999)