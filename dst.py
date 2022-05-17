from typing import Any, Callable

import time
import torch
from torch.serialization import save
import torch.utils.data as Data
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import time
import torch.distributed as dist
import os
import socket
import loader
import common

GPU_NUM= 2
LOG_HOST = "myri-gw2"

def set_gpu(rank:int):
    torch.cuda.set_device(rank%GPU_NUM)

def check_correct_rate(net, test_loader):
    correct_num = 0

    net.cuda()
    net.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            b_x = x.cuda()
            b_y = y.cuda()
            output = net(b_x)
            prey_y = torch.max(output, 1)[1].cuda().data
            correct_num += torch.sum(prey_y == b_y).type(torch.FloatTensor)
            if step%50==0:
                print("{}/{}...".format(step, len(test_loader)))
    net.train()

    return float(correct_num/len(test_loader.dataset))

def get_layers(net:nn.Module):
    def _search_layers(net:nn.Module, layers:list):
        for l in net.children():
            if type(l) is nn.Sequential:
                _search_layers(l, layers)
            elif type(l) in [nn.Conv2d, nn.Linear]:
                layers.append(l)
    layers = []
    _search_layers(net, layers)
    return layers

class LogClient():
    def __init__(self, host:str, port:int) -> None:
        self.host = host
        self.port = port
        self.s = socket.socket()

    def conn(self):
        print("Connect to {}:{}...".format(self.host, self.port))
        self.s.connect((self.host, self.port))
    
    def send(self, message:str):
        self.s.sendall(message.encode())
    
    def close(self):
        self.s.close()

class Scheduler():
    def __init__(self, global_bs, base_bs, scaling='none', start_lr=1e-4, tot_steps=1000, end_lr=0., warmup_steps=0) -> None:
        self.global_bs = global_bs
        self.base_bs = base_bs
        self.scaling = scaling
        self.start_lr = start_lr
        self.tot_steps = tot_steps
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps

    def __call__(self, optimizer, iternum):
        if self.scaling=='sqrt':
            init_lr = np.sqrt(self.global_bs/self.base_bs)*self.start_lr
        elif self.scaling=='linear':
            init_lr = (self.global_bs/self.base_bs)*self.start_lr
        elif self.scaling=='none':
            init_lr = self.start_lr
        elif self.scaling=='null':
            optimizer.param_groups[0]['lr'] = self.start_lr
            return

        if self.global_bs > self.base_bs and self.scaling != 'none':
            # warm-up lr rate
            if iternum<self.warmup_steps:
                lr = (iternum/self.warmup_steps)*init_lr
            else:
                lr = self.end_lr + 0.5 * (init_lr - self.end_lr) * (1 + np.cos(np.pi * (iternum - self.warmup_steps)/self.tot_steps))
        else:
            lr = self.end_lr + 0.5 * (init_lr - self.end_lr) * (1 + np.cos(np.pi * iternum/self.tot_steps))
        optimizer.param_groups[0]['lr'] = lr

def train(net, 
            train_loader, 
            test_loader, 
            loss_func, 
            optimizer,
            epochs, 
            save_name, 
            scheduler:Scheduler, 
            early_stop: float=None, 
            steps_stop:int=None,
            is_only_valid:bool=False,
            rank=0, 
            after_backward_hook: Callable = None, 
            log_server_port:int=19999, 
            **hook_args: Any):
    def print_overwrite(step, goal, running_loss, status):
        return "{}/{},  {} loss={}...".format(step, goal, status, running_loss )
    def cluster_print(msg:str, rank:int, client:LogClient):
        print(msg)
        if rank == 0:
            client.send(msg)
    
    client = LogClient(LOG_HOST, log_server_port)
    if rank == 0:
        client.conn()
    cluster_print("Start trainning...", rank, client)
    loss_last = np.inf
    acc_max = 0
    net.cuda()
    for epoch in range(epochs):
        loss_valid = 0
        loss_train = 0
        net.train()
        for step, (x, y) in enumerate(train_loader):
            if is_only_valid:
                break
            b_x = x.cuda()
            b_y = y.cuda()
            output = net(b_x)
            loss = loss_func(output, b_y)
            scheduler(optimizer, epoch)
            optimizer.zero_grad()
            loss.backward()
            if after_backward_hook:
                after_backward_hook(net.module, **hook_args, step=step)
            optimizer.step()
            loss_train += loss.item()
            runtime_loss = loss_train/(step+1)
            if step%50 == 0:
                cluster_print(print_overwrite(step, len(train_loader), runtime_loss, 'train'), rank, client)
            if steps_stop:
                if step == steps_stop:
                    break
        correct_num = torch.sum(torch.zeros(1)).cuda()
        net.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                b_x = x.cuda()
                b_y = y.cuda()
                output = net(b_x)
                loss = loss_func(output, b_y)
                loss_valid += loss.item()
                prey_y = torch.max(output, 1)[1].cuda().data
                new_correct_num = torch.sum(prey_y == b_y).type(torch.FloatTensor).cuda()
                dist.all_reduce(new_correct_num)
                correct_num += new_correct_num
                runtime_loss = loss_valid/(step+1)
                if step%50 == 0:
                    cluster_print(print_overwrite(step, len(test_loader), runtime_loss, 'val'), rank, client)
                

        loss_train /= len(train_loader)
        loss_valid /= len(test_loader)
        correct_num = correct_num.cpu()
        accuracy = correct_num/len(test_loader.dataset)

        msg = ""
        msg += '\n--------------------------------------------------'
        msg += 'Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}  Correct Rate: {:.4f}'.format(epoch, loss_train, loss_valid, accuracy)
        msg += '--------------------------------------------------'
        cluster_print(msg, rank, client)

        if is_only_valid:
            break

        if early_stop:
            diff = torch.tensor(loss_last - loss_valid).cuda()
            dist.all_reduce(diff, op=dist.ReduceOp.MAX)
            if diff.item() < early_stop:
                msg = "The reduction of valid loss is less than {}, stop traning.\n".format(early_stop)
                msg += 'Model Saved\n'
                if rank == 0:
                    torch.save(net.module.state_dict(), save_name) 
                cluster_print(msg, rank, client)
                break

        loss_last = loss_valid

        if acc_max < accuracy:
            acc_max = accuracy
            if rank == 0:
                torch.save(net.module.state_dict(), save_name) 
            msg = ""
            # print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, EPOCH))
            msg += "\nMaximum Validation Accuracy of {:.4f} at epoch {}/{}".format(accuracy, epoch, epochs)
            msg += 'Model Saved\n'
            cluster_print(msg , rank, client)
        
    if rank == 0:
        client.send("[STOP]")
        client.close()

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))

def distributed_train(net: torch.nn.Module, 
                    load_name: str = None, 
                    save_name: str = "trained_model", 
                    dataset:str="ImageNet", 
                    epochs:int=60,
                    steps_stop:int=None,
                    early_stop:float=None, 
                    per_worker_batch_size:int=32, 
                    lr:float=0.01, 
                    scheduler_type:str="none",
                    is_only_valid:bool=False,
                    worker_nums:int=4, 
                    master:str="c2",
                    log_server_port:int=19999, 
                    find_unused_parameters:bool=False, 
                    after_backward_hook:Callable=None, 
                    **hook_args:Any):
    rank = int(os.environ["SLURM_PROCID"])

    WOKER_NUMS = worker_nums
    MASTER = master
    EPOCH = epochs
    PER_WORKER_BATCH = per_worker_batch_size
    LR = lr

    print("[RANK{}]Initialize distributed environment...".format(rank))
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, init_method="tcp://{}:8989".format(MASTER), world_size=WOKER_NUMS)
    torch.cuda.set_device(rank%GPU_NUM)

    # transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    # transforms.ConvertImageDtype(torch.float),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("[RANK{}]Initialize dataloader...".format(rank))
    
    if dataset == "ImageNet":
        # train_set = loader.ImageNetDataset(transform=transform)
        # test_set = loader.ImageNetDataset(val=True, transform=transform)
        train_set = datasets.ImageNet(root="/home/dataset/imagenet", split="train", transform=transform)
        test_set = datasets.ImageNet(root="/home/dataset/imagenet", split="val", transform=transform)
    elif dataset == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    
    train_sampler = Data.DistributedSampler(train_set)
    train_loader = Data.DataLoader(train_set, batch_size=PER_WORKER_BATCH, sampler=train_sampler, num_workers=4, worker_init_fn=worker_init, persistent_workers=True, pin_memory=torch.cuda.is_available())
    test_sampler = Data.DistributedSampler(test_set)
    test_loader = Data.DataLoader(test_set, batch_size=PER_WORKER_BATCH, sampler=test_sampler, num_workers=4, worker_init_fn=worker_init, pin_memory=torch.cuda.is_available())

    if load_name and rank==0:
        print("Load from {}...".format(load_name))
        net.load_state_dict(torch.load(load_name))
    net.cuda()
    ddp_net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank%GPU_NUM], find_unused_parameters=find_unused_parameters)

    scheduler = Scheduler(WOKER_NUMS*PER_WORKER_BATCH, PER_WORKER_BATCH, scheduler_type, LR, EPOCH, 0, 0)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss()

    start_time = time.process_time()
    print("Start training...")
    hook_args["total_steps"] = len(train_loader)
    hook_args["worker_nums"] = WOKER_NUMS
    train(ddp_net, 
        train_loader, 
        test_loader, 
        loss_func, 
        optimizer, 
        EPOCH, 
        save_name, 
        scheduler, 
        rank=rank,
        steps_stop=steps_stop,
        early_stop=early_stop,
        is_only_valid=is_only_valid,
        after_backward_hook=after_backward_hook, 
        log_server_port=log_server_port, 
        **hook_args)
    print("training time = {}".format(time.process_time() - start_time))

if __name__ == "__main__":
    
    net = models.resnet34()
    common.add_masks(net)

    train_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    model_path = "checkpoint/RestNet/resnet34/16_16/untuned/ResNet_MetricsL1_TW_pruning_ReconstuctIsTrue_ReconMethodL1Sort_{}"

    for sp in train_list:
        model_name = model_path.format(sp)
        save_path = "work_space/" + model_name.split('/')[-1]
        distributed_train(net, 
                        load_name=model_name, 
                        save_name=save_path, 
                        epochs=10,
                        steps_stop=None,
                        early_stop=0.01,
                        per_worker_batch_size=64, 
                        lr=0.01,
                        scheduler_type="null",
                        is_only_valid=False,
                        dataset="ImageNet", 
                        worker_nums=8, 
                        log_server_port=19999,
                        master="c2",
                        find_unused_parameters=True)