import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image
import os
import random


class CifarDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform:
            data = self.transform(self.data[idx])
        else:
            data = torch.from_numpy(self.data[idx]).float()
        sample = [data,  torch.tensor(self.labels[idx]).long()]
        
        return sample
        

class ImageNetDataset(Dataset):
    def load(self, val=False):
        if val:
            data_path = "/home/dataset/imagenet/ILSVRC2012_val"
        else:
            data_path = "/home/dataset/imagenet/ILSVRC2012"
        labels = []
        images = []
        for _, dirs, _ in os.walk(data_path):
            for label, dir in enumerate(dirs):
                for root, _, files in os.walk(os.path.join(data_path, dir)):
                    for f in files:
                        labels.append(label)
                        images.append(os.path.join(root, f))
        return images, labels

    def __init__(self, val=False, transform=None, worker_nums=1, rank=0):
        self.transform = transform
        self.worker_nums = worker_nums
        self.rank = rank
        self.images, self.labels = self.load(val=val)
        self.origin = (len(self.labels)//worker_nums)*rank
    
    def __len__(self):
        return len(self.labels)//self.worker_nums
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        if type(index) is int:
            index += self.origin
        elif type(index) is slice:
            index = slice(index.start+self.origin, index.stop+self.origin, index.step+self.origin)
        else:
            for i in range(len(index)):
                index[i] += self.origin
        while True:
            try:
                image = read_image(self.images[index], torchvision.io.ImageReadMode.RGB)
                break
            except Exception as e:
                print("Error ocurred in index:{}! ".format(index))
                print(str(e))
                index = random.randint(0, len(self.labels))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label
        

def load_data(filename, transform=None):
    label = []
    data = []
    with open("cifar-10-batches-py/" + filename, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        for i in range(len(dict[b"data"])):
            d = np.frombuffer(dict[b"data"][i], dtype=np.uint8).reshape(32,32,3)
            data.append(Image.fromarray(d))
            label.append(dict[b"labels"][i])
    label = np.array(label)
    dataset = CifarDataset(data, label, transform=transform)
    return dataset


# transform = transforms.Compose([
#     # transforms.Resize(256),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# d = load_data("test_batch", transform=transform)

# td = transform(d.data[0])

# print(td)
# print(td.shape)

# test_loader = DataLoader(dataset=d, batch_size=128, shuffle=True)
# for step, (x, y) in enumerate(test_loader):
#     print('t')