import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import sys
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import models.models_sparse.layers as nn_flags
import loader

def check_correct_rate(net, test_loader):
    correct_num = 0

    net.cuda()
    net.eval()
    with torch.no_grad():
        print_interval_steps = len(test_loader)//10
        for step, (x, y) in enumerate(test_loader):
            b_x = x.cuda()
            b_y = y.cuda()
            output = net(b_x)

            prey_y = torch.max(output, 1)[1].cuda().data
            correct_num += torch.sum(prey_y == b_y).type(torch.FloatTensor)
            if step%print_interval_steps==0:
                print("{}/{}...".format(step, len(test_loader)))
    net.train()

    return float(correct_num/len(test_loader.dataset))


def print_model_weight(net):
    for layer in net.children():
        if type(layer) is nn.Sequential:
            print_model_weight(layer)
        else:
            if type(layer) is nn.Conv2d:
                print(str(layer) + ' sparsity: ' +  str(float(torch.sum(layer.weight == 0))/float(layer.weight.nelement())))
            elif type(layer) is nn.Linear:
                print(str(layer) + ' sparsity: ' +  str(float(torch.sum(layer.weight == 0))/float(layer.weight.nelement())))


def get_layers(net:nn.Module):
    def _search_layers(net:nn.Module, layers:list):
        for l in net.children():
            if type(l) in [nn.Sequential, models.resnet.BasicBlock]:
                _search_layers(l, layers)
            # elif type(l) in [nn.Conv2d, nn.Linear, nn_flags.Conv_flags, nn_flags.Linear_flags]:
            elif type(l) in [nn.Conv2d, nn.Linear]:
                layers.append(l)
    layers = []
    _search_layers(net, layers)
    return layers


def imply_mask(self, input):
    with torch.no_grad():
        self.weight = nn.Parameter(self.weight*self.mask)

def add_masks(net):
    layers = get_layers(net)
    with torch.no_grad():
        masks = [torch.ones(layer.weight.shape) for layer in layers]
        for i in range(len(masks)):
            layers[i].register_buffer('mask', masks[i])
            layers[i].register_forward_pre_hook(imply_mask)


def padding(weight, tile_shape):
    weight_2d = weight.view((-1,-1))
    W = tile_shape[0]
    H = tile_shape[1]
    w_to_padding = weight_2d.shape[0] % W
    h_to_padding = weight_2d.shape[1] % H
    padded_weight = torch.cat((weight_2d, torch.zeros((weight_2d.shape[0], w_to_padding))), 1)
    padded_weight = torch.cat((padded_weight, torch.zeros((h_to_padding, padded_weight.shape[1]))), 0)
    return padded_weight


def shifting(weight, value, dim):
    weight_left, weight_right = torch.split(weight, (weight.shape[dim] - value, value), dim)
    return torch.cat((weight_right, weight_left), dim)


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

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))

def check_ImangeNet_model_accuracy(net:torch.nn.Module, batch_size:int=16) -> float:
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_set = loader.ImageNetDataset(val=True, transform=transform)
    test_loader = Data.DataLoader(test_set, batch_size=batch_size, num_workers=4, worker_init_fn=worker_init, pin_memory=torch.cuda.is_available())
    return check_correct_rate(net, test_loader)

def get_dataset(dataset:str="ImageNet"):
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if dataset == "ImageNet":
        # train_set = loader.ImageNetDataset(transform=transform)
        # test_set = loader.ImageNetDataset(val=True, transform=transform)
        train_set = datasets.ImageNet(root="/home/dataset/imagenet", split="train", transform=transform)
        test_set = datasets.ImageNet(root="/home/dataset/imagenet", split="val", transform=transform)
    elif dataset == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    return train_set, test_set