from pyexpat import model
from typing import List

from webbrowser import get
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import random
import time
import types
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from dst import get_layers

from models.models_sparse.layers import construct as sparse_construct
from models.models_sparse.layers import free as sparse_free
from models.models_dense.layers import AlexNet_cutlass
from models.models_dense.layers import free as dense_free
import common
import reconstructor
import metrics
import pruner

def read_log_0(filename):
    acc_list = []
    with open(filename) as f:
        for l in f:
            if "Maximum Validation Accuracy of" in l:
                acc_list.append(float(l.split(" ")[4]))
    print(acc_list)

def read_log_1(filename):
    acc_list = []
    with open(filename) as f:
        for l in f:
            if "Correct Rate: " in l:
                acc = float(l.split(' ')[-1].split('-')[0])
            if "training time =" in l:
                acc_list.append(acc)
    print(acc_list)

def test_recover():
    a = torch.abs(torch.randn((8,8)))
    line_sum = []
    for i in range(a.shape[0]):
        line_sum.append(torch.sum(a[i]))
    line_sum = torch.Tensor(line_sum)
    _, index = torch.sort(line_sum)
    print(a)
    print(index)
    b = a.index_select(0, index)
    print(b)
    recover_index = [0 for i in range(len(index))]
    for i in range(len(index)):
        recover_index[index[i]] = i
    recover_index = torch.IntTensor(recover_index)
    print(recover_index)
    c = b.index_select(0, recover_index)
    print(c)
    print(a == c)

def weight_recover(weight, row_transform, column_transform, is_change = False):
    index = row_transform
    recover_index = [0 for i in range(len(index))]
    for i in range(len(index)):
        recover_index[index[i]] = i
    recover_index = torch.IntTensor(recover_index)
    weight = weight.index_select(0, recover_index)
    index = column_transform
    if is_change:
        changed_index = torch.arange(weight.shape[1])
        changed_index = changed_index.view((len(index), -1))
        index = changed_index.index_select(0, index).view(-1)
    recover_index = [0 for i in range(len(index))]
    for i in range(len(index)):
        recover_index[index[i]] = i
    recover_index = torch.IntTensor(recover_index)
    weight = weight.index_select(1, recover_index)
    return weight

def cal_similarity(weights0, weights1, ty="cos"):
    if ty=="cos":
        cos = nn.CosineSimilarity(dim=0)
        similarity = cos(weights0, weights1)
    elif ty=="l1":
        p0 = torch.abs(weights0)
        p1 = torch.abs(weights1)
        diff = p0 - p1
        similarity = diff.sum()
    return similarity


def cat_all_weights(net, recon_list_list=[]):
    layers = common.get_layers(net)
    if len(recon_list_list) > 0:
        weights = [l.weight for l in layers]
        for j in range(len(recon_list_list)-1, -1, -1):
            recon_list = recon_list_list[j]
            for i in range(len(layers)):
                weight = weights[i]
                if i == len(layers)-1:
                    row_trans = torch.arange(weight.shape[0])
                else:
                    row_trans = recon_list[i]
                if i == 0:
                    col_trans = torch.arange(weight.shape[1])
                else:
                    col_trans = recon_list[i-1]
                if i>0:
                    if type(layers[i]) is nn.Linear and type(layers[i-1]) is nn.Conv2d:
                        weight = weight_recover(weight, row_trans, col_trans, True)
                    else:
                        weight = weight_recover(weight, row_trans, col_trans)
                else:
                    weight = weight_recover(weight, row_trans, col_trans)
                weights[i] = weight
        weights_1d = [w.view((1,-1)) for w in weights]
    else:
        weights_1d = [ l.weight.view((1,-1)) for l in layers ]
    out = torch.cat(weights_1d, 1).view(-1)
    return out
        
def check_similarity(is_recon=False, sparsity=0.5, ty="cos"):
    SPARSITY = sparsity
    recon = reconstructor.Reconstructor(metrics=metrics.MetricsL1, method=reconstructor.ReconMethodL1Sort)
    random_input = torch.randn((1, 3, 224, 224))
    
    net0 = models.alexnet(pretrained=True)
    pruner.oneshot_prune(net0, method=pruner.EW_pruning, sparsity=SPARSITY)
    net0(random_input)

    net1 = models.alexnet(pretrained=True)
    if is_recon:
        recon_list = recon(net1)
    pruner.oneshot_prune(net1, method=pruner.TW_pruning, tile_shape=[32, 32], sparsity=SPARSITY)
    net1(random_input)

    weights0 = cat_all_weights(net0)
    if is_recon:
        weights1 = cat_all_weights(net1, [recon_list])
    else:
        weights1 = cat_all_weights(net1)
    similarity = cal_similarity(weights0, weights1, ty=ty)
    return similarity

def check_steps_similarity(is_recon=False, sparsity_per_step=0.02, ty="cos"):
    net0 = models.alexnet(pretrained=True)
    net1 = models.alexnet(pretrained=True)
    ns = 0
    recon_list_list = []
    s_list = []
    while ns < 1:
        recon = reconstructor.Reconstructor(metrics=metrics.MetricsL1, method=reconstructor.ReconMethodL1Sort)
        random_input = torch.randn((1, 3, 224, 224))
        
        pruner.oneshot_prune(net0, method=pruner.EW_pruning, sparsity=ns)
        net0(random_input)
        
        if is_recon:
            recon_list = recon(net1)
            recon_list_list.append(recon_list)
        pruner.oneshot_prune(net1, method=pruner.TW_pruning, tile_shape=[32, 32], sparsity=ns)
        net1(random_input)

        weights0 = cat_all_weights(net0)
        if is_recon:
            weights1 = cat_all_weights(net1, recon_list_list)
        else:
            weights1 = cat_all_weights(net1)
        similarity = cal_similarity(weights0, weights1, ty=ty)
        s_list.append(similarity.item())
        ns += sparsity_per_step
    return s_list


def get_layer_sparsity(net:nn.modules, model_path:str):
    pruner.add_masks(net)
    sparsities = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    layer_sparisty = []
    for s in sparsities:
        net.load_state_dict(torch.load(model_path.format(s)))
        layer_sparisty.append(common.check_model_saprsity(net))
    return layer_sparisty


def cal_model_loss(net:nn.modules, sparsity:float, tile_shape:List, is_recon:bool=False):
    random_input = torch.randn((1, 3, 224, 224))

    # method = pruner.EW_pruning(metrics.MetricsL1)
    method = pruner.TW_pruning(tile_shape ,metrics.MetricsL1)
    p = pruner.Pruner(method)
    

    if is_recon:
        recon = reconstructor.Reconstructor(metrics=metrics.MetricsL1, method=reconstructor.ReconMethodL1Sort)
        recon(net)

    p.prune(net, sparsity)
    net(random_input)
    # common.check_model_saprsity(net)
    
    layers = common.get_layers(net)
    s = torch.Tensor([0])
    for l in layers:
        s += torch.sum(torch.abs(l.weight))
    return s
    
    


if __name__ == '__main__': 

    def test_conv():
        feed = torch.randn((1, 256, 224, 224))

        net = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
        )

        feed_d = feed.cuda()
        net.cuda()
        out0 = net(feed_d)
        
        net.cpu()
        sparse_construct(net)
        net.cuda()
        out1 = net(feed_d)

        print(torch.sum(out0-out1))

    def test_linear():
        feed = torch.randn((1024, 1024))

        net = nn.Sequential(
            nn.Linear(1024, 100)
        )

        feed_d = feed.cuda()
        net.cuda()
        out0 = net(feed_d)
        
        net.cpu()
        sparse_construct(net)
        net.cuda()
        out1 = net(feed_d)

        print(out0.shape)
        print(out1.shape)
        print(torch.sum(out0-out1))

    time_list = []

    for i in range(1,2):
        net = models.alexnet(pretrained=True)
        pruner.add_masks(net)
        # net.load_state_dict(torch.load("checkpoint/AlexNet/ImageNet/fine_tuned/128_1/l1/AlexNet_MetricsL1_TW_pruning_ReconstuctIsFalse_ReconMethodL1Sort_10"))
        # net.load_state_dict(torch.load("checkpoint/IMAGENET/ResNet_MetricsL1_TW_pruning_ReconstuctIsTrue_ReconMethodL1Sort_10".format(i*10)))
        sparse_construct(net)
        net.cuda()
        
        IMAGE_NUMBER = 10000

        start_time = time.process_time()

        for i in range(IMAGE_NUMBER):
            test_data = torch.randn((1,3,224,224))
            net(test_data.cuda())
            if i%500 == 0:
                print(i)
        
        interval = time.process_time() - start_time
        print(interval/IMAGE_NUMBER)
        time_list.append(interval/IMAGE_NUMBER)

    print(time_list)