from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import pretrainedmodels
import time
import glob
import os
from transforms import *
from torch import nn
import models


class AtlasData(Dataset):
    def __init__(self, split, train = True, model = 'bninception'):
        self.split = split
        self.train = train
        self.train_str = 'train' if self.train else 'test'
        self.text_file = 'data/atlas_{}_split_{}.txt'.format(self.train_str, self.split)
        
        self.data = [[y for y in x.strip().split(' ')] for x in open(self.text_file, 'r').readlines()]
        self.imgs = [x[0] for x in self.data]
        self.labels = [[int(p) for p in x[1:]] for x in self.data]
        
        self.input_size = models.model_configs[model]['input_size']
        self.input_mean = models.model_configs[model]['input_mean']
        self.input_std = models.model_configs[model]['input_std']

        
        self.transforms = transforms.Compose([GroupRandomRotate(360),
                                              GroupScale(self.input_size),
                                              Stack(roll=False),
                                              ToTorchFormatTensor(div=True),
                                              transforms.Normalize(self.input_mean, self.input_std),
                                            ])
       
        if train == False:
            self.transforms = transforms.Compose([
                                              GroupScale(self.input_size),
                                              Stack(roll=False),
                                              ToTorchFormatTensor(div=True),
                                              transforms.Normalize(self.input_mean, self.input_std),
                                            ])
       

    
    
    def load_image_stack(self, image_id):
        colors = ['red', 'green', 'blue', 'yellow']
        absolute_paths = ["data/train/{}_{}.png".format(image_id, color) for color in colors]
        images = [Image.open(path).convert('L') for path in absolute_paths]
        
        return images
    
    def __getitem__(self, i):
        image_id = self.imgs[i]
        image = self.load_image_stack(image_id)
        image = self.transforms(image)
        
        label = self.labels[i]
        label_arr = np.zeros(28, dtype = np.float32)
        [np.put(label_arr, x, 1) for x in label]
        
        label_arr = torch.from_numpy(label_arr)
        
        return image, label_arr, label
        
    def __len__(self):
        return len(self.imgs)

class EvalAtlasData(Dataset):
    def __init__(self, model):
        self.image_ids = sorted(set([x.split('_')[0] for x in os.listdir('data/test')]))
        
        self.input_size = models.model_configs[model]['input_size']
        self.input_mean = models.model_configs[model]['input_mean']
        self.input_std = models.model_configs[model]['input_std']
        
        self.transforms = transforms.Compose([GroupScale(self.input_size),
                                              Stack(roll=False),
                                              ToTorchFormatTensor(div=True),
                                              transforms.Normalize(self.input_mean, self.input_std),
                                            ])
        
    def load_image_stack(self, image_id):
        colors = ['red', 'green', 'blue', 'yellow']
        absolute_paths = ["data/test/{}_{}.png".format(image_id, color) for color in colors]
        images = [Image.open(path).convert('L') for path in absolute_paths]
        
        return images
    
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, i):
        image_id = self.image_ids[i]
        images = self.load_image_stack(image_id)
        image = self.transforms(images)
            
        return image_id, image