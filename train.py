import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import sys
import torchvision.models as models
import torchvision.transforms as transforms
import time
import os

import common

GPU_NUM=2

import loader
from models.models_torch.layers import AlexNet

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



def train(net, train_loader, test_loader, loss_func, optimizer, epochs, save_name):
    def print_overwrite(step, goal, running_loss, status):
        print("{}/{},  {} loss={}...".format(step, goal, status, running_loss ))

    loss_min = np.inf
    acc_max = 0
    net.cuda()

    for epoch in range(epochs):

        loss_valid = 0
        loss_train = 0
        runtime_loss = 0

        net.train()

        for step, (x, y) in enumerate(train_loader):

            b_x = x.cuda()
            b_y = y.cuda()

            output = net(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            runtime_loss += loss_train/(step+1)

            # if step%50 == 0:
            # print_overwrite(step, len(train_loader), runtime_loss, 'train')
            if step == 10:
                break

        correct_num = 0

        net.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                b_x = x.cuda()
                b_y = y.cuda()
                output = net(b_x)
                loss = loss_func(output, b_y)
                loss_valid += loss.item()

                prey_y = torch.max(output, 1)[1].cuda().data
                correct_num += torch.sum(prey_y == b_y).type(torch.FloatTensor)
                
                # print_overwrite(step, len(test_loader), runtime_loss, 'valid')
                break


        loss_train /= len(train_loader)
        loss_valid /= len(test_loader)

        accuracy = correct_num/len(test_loader.dataset)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}  Correct Rate: {:.4f}'.format(epoch, loss_train, loss_valid, accuracy))
        print('--------------------------------------------------')

        # if loss_valid < loss_min:
        if acc_max < accuracy:
            acc_max = accuracy
            torch.save(net.state_dict(), save_name) 
            # print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, EPOCH))
            print("\nMaximum Validation Accuracy of {:.4f} at epoch {}/{}".format(accuracy, epoch, epochs))
            print('Model Saved\n')

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))

if __name__ == '__main__':
    
    WOKER_NUMS=2
    EPOCH = 1          
    BATCH_SIZE = 256
    LR = 0.01
    
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = loader.ImageNetDataset(transform=transform)
    train_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, worker_init_fn=worker_init, persistent_workers=True, pin_memory=torch.cuda.is_available())
    test_set = loader.ImageNetDataset(val=True, transform=transform)
    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=4, worker_init_fn=worker_init, pin_memory=torch.cuda.is_available())

    net = models.alexnet(pretrained=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss()

    start_time = time.process_time()
    train(net, train_loader, test_loader, loss_func, optimizer, EPOCH, "trained_model")
    print("training time = {}".format(time.process_time() - start_time))
    
    layers = common.get_layers(net)
    print(type(layers[0].weight.grad))
    
    layers[0].weight.grad = torch.zeros_like(layers[0].weight)