from readline import insert_text
from tkinter import image_names
import numpy as np
from PIL import Image
import random
from torchvision.transforms import InterpolationMode
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2 


class Compose(object):
    def __init__(self, transforms):
        self.transforms=transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image,target


class Resize(object):
    def __init__(self,output_size=384,train=True) -> None:
        self.size=output_size
        self.train=train

    def __call__(self, image,target):
        image = F.resize(image, (self.size,self.size))
        # we must need to test on the original size 
        if self.train:
            target = F.resize(target, (self.size, self.size), interpolation=InterpolationMode.NEAREST)
         
        return image, target




class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.tensor(np.asarray(target), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        # print(image.shape, '.,..')
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


# We don't apply other complex data augment
def get_transform(size, train=True):
    transforms = []
    transforms.append(Resize(size, train))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return Compose(transforms)

