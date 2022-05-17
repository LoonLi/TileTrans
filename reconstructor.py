from typing import Union, List, Dict, Any

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data as Data
import common
from metrics import Metrics, MetricsL1, MetricsGradient
import dst
from itertools import zip_longest

class ReconMethod():
    pass

# This method reconstruct the weight matrix by the L1 distance of the line vector.
# In detail, it calculate the sum of the absolute value of each line of the evaluated matrix, then sort the lines based on the sum value. 
# The metrics to evalute the matrix can be assigned mannually. 
class ReconMethodL1Sort(ReconMethod):
    @staticmethod    
    def sort(metrics:Metrics = MetricsL1, layers_group: List[torch.nn.Module] = None, grads_group:List[torch.Tensor] = [], **kargs: Any) -> torch.Tensor:
        if layers_group is None:
            raise ValueError("Layer must be {}".format(torch.nn.Module))
        to_cat = []
        for layer, grad in zip_longest(layers_group, grads_group):
            weight = metrics.eval(layer, grad=grad)
            weight = weight.view((weight.shape[0], -1))
            to_cat.append(weight)
        if len(to_cat) > 1:
            weight = torch.cat(to_cat, dim=1)
        else:
            weight = to_cat[0]
        line_sum = []
        for i in range(weight.shape[0]):
            line_sum.append(torch.abs(weight[i]).sum())
        line_sum = torch.Tensor(line_sum)
        _, index = torch.sort(line_sum)
        return index


class Reconstructor():
    def __init__(self, metrics: Metrics = MetricsL1, method: ReconMethod = ReconMethodL1Sort) -> None:
        self.metrics = metrics
        self.method = method
    
    def __call__(self, net: torch.nn.Module = None, grads: List[torch.Tensor] = None, **kargs:Any) -> List:
        if net is None:
            raise ValueError("Net must be {}".format(torch.nn.Module))

        def transform_rows(layer:nn.modules, index:torch.Tensor) -> None :
            reconstructed_weight = torch.clone(layer.weight).index_select(0, index)
            layer.weight = nn.Parameter(reconstructed_weight)
            if not (layer.bias is None):
                reconstructed_bias = torch.clone(layer.bias).index_select(0, index)
                layer.bias = nn.Parameter(reconstructed_bias)
        
        def transform_cols(layer:nn.modules, index:torch.Tensor) -> None :
            reconstructed_weight = torch.clone(layer.weight).index_select(1, index)
            layer.weight = nn.Parameter(reconstructed_weight)

        def reconstruct_by_index(layer_front, layer_back, index):
            with torch.no_grad():
                reconstructed_weight = torch.clone(layer_front.weight).index_select(0, index)
                reconstructed_bias = torch.clone(layer_front.bias).index_select(0, index)
                layer_front.weight = nn.Parameter(reconstructed_weight)
                layer_front.bias = nn.Parameter(reconstructed_bias)

                if type(layer_front) is nn.Conv2d and type(layer_back) is nn.Linear:
                    changed_index = torch.arange(layer_back.weight.shape[1]).view((len(index), -1))
                    changed_index = changed_index.index_select(0, index).view(-1)
                    reconstructed_weight = torch.clone(layer_back.weight).index_select(1, changed_index)
                    layer_back.weight = nn.Parameter(reconstructed_weight)
                else:
                    reconstructed_weight = torch.clone(layer_back.weight).index_select(1, index)
                    layer_back.weight = nn.Parameter(reconstructed_weight)

        def reconstruct_grads_by_index(layer_front, layer_back, grads, i, index):
            with torch.no_grad():
                reconstructed_grads = torch.clone(grads[i].cpu()).index_select(0, index)
                grads[i] = reconstructed_grads
                
                if type(layer_front) is nn.Conv2d and type(layer_back) is nn.Linear:
                    changed_index = torch.arange(layer_back.weight.shape[1]).view((len(index), -1))
                    changed_index = changed_index.index_select(0, index).view(-1)
                    reconstructed_grads = torch.clone(grads[i+1].cpu()).index_select(1, changed_index)
                    grads[i+1] = reconstructed_grads
                else:
                    reconstructed_grads = torch.clone(grads[i+1].cpu()).index_select(1, index)
                    grads[i+1] = reconstructed_grads
                
        if type(net) in [models.AlexNet, models.VGG]:

            recon_list = []
            layers = common.get_layers(net)
            for i in range(len(layers)-1):
                layer_0 = layers[i]
                layer_1 = layers[i+1]
                if grads:
                    index = self.method.sort(self.metrics, [layer_0], [grads[i]])
                    reconstruct_by_index(layer_0, layer_1, index)
                    reconstruct_grads_by_index(layer_0, layer_1, grads, i, index)
                else:
                    index = self.method.sort(self.metrics, [layer_0])
                    reconstruct_by_index(layer_0, layer_1, index)
                recon_list.append(index)
            return recon_list

        elif type(net) is models.ResNet:
            
            recon_dic = {}

            for c in net.children():
                if type(c) is nn.Sequential:
                    for b in c.children():
                        if type(b) is models.resnet.BasicBlock:
                            index = self.method.sort(self.metrics, [b.conv1])
                            transform_rows(b.conv1, index)
                            transform_rows(b.bn1, index)
                            transform_cols(b.conv2, index)
            
            layers = [net.layer1, net.layer2, net.layer3, net.layer4]
            layers_row_group = []
            layers_col_group = []
            bns_group = []
            layers_row_group.append(net.conv1)
            bns_group.append(net.bn1)
            for i,l in enumerate(layers):
                for j,b in enumerate(l.children()):
                    if i == 0 and j==0:
                        layers_col_group.append(b.conv1)
                    elif j!= 0:
                        layers_col_group.append(b.conv1)
                    if not (b.downsample is None):
                        layers_row_group.append(b.downsample._modules['0'])
                        bns_group.append(b.downsample._modules['1'])
                    layers_row_group.append(b.conv2)
                    bns_group.append(b.bn2)
                if i == 3:
                    layers_col_group.append(net.fc)
                else:
                    layers_col_group.append(layers[i+1]._modules['0'].conv1)
                    layers_col_group.append(layers[i+1]._modules['0'].downsample._modules['0'])
                index = self.method.sort(self.metrics, layers_row_group)
                for trans_layer in layers_row_group:
                    transform_rows(trans_layer, index)
                for bn in bns_group:
                    transform_rows(bn, index)
                for recover_layer in layers_col_group:
                    if type(recover_layer) is nn.Linear:
                        changed_index = torch.arange(net.layer4._modules['2'].conv2.weight.shape[1]).view((len(index), -1))
                        changed_index = changed_index.index_select(0, index).view(-1)
                        transform_cols(recover_layer, changed_index)
                    else:
                        transform_cols(recover_layer, index)
                layers_row_group = []
                layers_col_group = []
                bns_group = []
            
            return recon_dic

if __name__ == "__main__":

    net = models.resnet34(pretrained=True)

    recon = Reconstructor()
    recon(net)
    dst.distributed_train(net, 
                    load_name=None, 
                    epochs=1,
                    steps_stop=50,
                    early_stop=0.01,
                    per_worker_batch_size=64, 
                    lr=0.001,
                    scheduler_type="null",
                    is_only_valid=False,
                    dataset="ImageNet", 
                    worker_nums=8, 
                    log_server_port=19998,
                    master="c5",
                    find_unused_parameters=True)