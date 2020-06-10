import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np

import os
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces

#The dataset for tiney imagenet 
class TineyImagenet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        
        root = os.path.expanduser(root)
        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
            
        self.target_transform = target_transform
            
        if train:
            self.data = np.load(root + '/train/img_train.npy')
            self.label = np.load(root + '/train/label_train.npy').astype(np.int64)
            self.label = torch.from_numpy(self.label)
            self.num = len(self.data)
        else:
            self.data = np.load(root + '/test/img_test.npy')
            self.label = np.load(root + '/test/label_test.npy').astype(np.int64)
            self.label = torch.from_numpy(self.label)
            self.num = len(self.data)
        
        self.data = np.uint8(self.data)
        
        
    def __len__(self):
        return self.num
        
        
    def __getitem__(self, idx):
        trans = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])
        out_data = trans(self.data[idx])
        out_label = self.label[idx]

        out_data = self.transform(out_data)
        
        if self.target_transform is not None:
            out_label = self.target_transform(out_label)

        return out_data, out_label

    
#The dataset for olivetti faces
class OlivettiFaces(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
            
        self.data = fetch_olivetti_faces(data_home = root)
        self.num = len(self.data.data)
        
    def __len__(self):
        return self.num
        
        
    def __getitem__(self, idx):
        out_data = self.transform(self.data.images[idx])
        out_label = self.data.target[idx]

        return out_data, out_label

class Glove(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        
        root = os.path.expanduser(root)
        
        if transform is None:
            self.transform = torch.tensor
        else:
            self.transform = transform
            
        self.data = np.load(root + '/data.npy')[:10000]
        f = open(root + '/label.txt')
        labels = f.readline()
        labels = labels.rstrip('\n').split(' ')
        self.labels = labels[:10000]
        self.num = len(self.data)
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        out_data = self.transform(self.data[idx])
        out_label = self.labels[idx]
        
        return out_data, out_label
    
    
    

class Datasets(object):
    def __init__(self, dataset_name, batch_size = 100, num_workers = 2, transform = None, shuffle = True):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.shuffle = shuffle
        
    def create(self):
        print("Dataset :",self.dataset_name)
        if self.transform is None:
                self.transform = transforms.Compose([transforms.ToTensor()])
        
        if self.dataset_name == "MNIST":
            trainset = torchvision.datasets.MNIST(root = '~/work/MNISTDataset/data',
                                       train = True, download = True, transform = self.transform)
            testset = torchvision.datasets.MNIST(root = '~/work/MNISTDataset/data',
                                                 train = False, download = True, transform = self.transform)
            classes = list(range(10))
            base_labels = trainset.classes
            
        elif self.dataset_name == "FashionMNIST":
            trainset = torchvision.datasets.FashionMNIST(root = '~/work/FashionMNISTDataset/data',
                                       train = True, download = True, transform = self.transform)
            testset = torchvision.datasets.FashionMNIST(root = '~/work/FashionMNISTDataset/data',
                                                 train = False, download = True, transform = self.transform)
            classes = list(range(10))
            base_labels = trainset.classes
            
        elif self.dataset_name == "CIFAR10":
            trainset = torchvision.datasets.CIFAR10(root = '~/work/CIFAR10Dataset/data',
                                       train = True, download = True, transform = self.transform)
            testset = torchvision.datasets.CIFAR10(root = '~/work/CIFAR10Dataset/data',
                                                 train = False, download = True, transform = self.transform)
            classes = list(range(10))
            base_labels = trainset.classes
            
        elif self.dataset_name == "CIFAR100":
            trainset = torchvision.datasets.CIFAR100(root = '~/work/CIFAR100Dataset/data',
                                       train = True, download = True, transform = self.transform)
            testset = torchvision.datasets.CIFAR100(root = '~/work/CIFAR100Dataset/data',
                                                 train = False, download = True, transform = self.transform)
            classes = list(range(100))
            base_labels = trainset.classes
        
        elif self.dataset_name == "TineyImagenet":            
            trainset = TineyImagenet(root = '~/work/TinyImagenet', train = True, transform = self.transform)
            testset = TineyImagenet(root = '~/work/TinyImagenet', train = False, transform = self.transform)
            
            classes = list(range(200))
            base_labels = []
            
        elif self.dataset_name == "OlivettiFaces":
            trainset = OlivettiFaces(root = '~/work/Olivettifaces/data', transform = self.transform)
            testset = None
            
            classes = list(range(40))
            base_labels = []
            
        elif self.dataset_name == "COIL-20":
            self.transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
            trainset = torchvision.datasets.ImageFolder(root='~/work/COIL-20/data', transform = self.transform)
            testset = None
            
            classes = list(range(20))
            base_labels = trainset.classes
        
        elif self.dataset_name == "Glove":
            trainset = Glove(root='~/work/Glove/data', transform = None)
            testset = None
            
            classes = list(range(len(trainset)))
            base_labels = trainset.labels
        
        
        
        else:
            raise KeyError("Unknown dataset: {}".format(self.dataset_name))
            
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = self.batch_size,
                        shuffle = self.shuffle, num_workers = self.num_workers)
        
        if testset is not None:
            testloader = torch.utils.data.DataLoader(testset, batch_size = self.batch_size,
                        shuffle = False, num_workers = self.num_workers)
        else:
            testloader = None
            
            
        return [trainloader, testloader, classes, base_labels, trainset, testset]
    
    def worker_init_fn(self, worker_id):                                                          
        np.random.seed(worker_id)