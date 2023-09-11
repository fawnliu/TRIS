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
# some transform will be used in referring image segmentation
# before to tensor, img is Pillow Image Object
# def pad_if_smaller(img,size,fill=0):
#     min_size=min(img.size)
#     if min_size < size:
#         ow,oh=img.size
#         padh = size - oh if oh < size else 0
#         padw = size - ow if ow < size else 0
#         img=F.pad(img,(0,0,padw,padh),fill=fill)
#     return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms=transforms
    
    def __call__(self, image, target, instance_masks=None):
        if instance_masks is not None:
            for t in self.transforms:
                image, target, instance_masks = t(image, target, instance_masks)
            return image, target, instance_masks 
        for t in self.transforms:
            image, target = t(image, target)
        return image,target

# 随机resize
# class RandomResize(object):
#     def __init__(self,min_size,max_size=None):
#         self.min_size=min_size
#         if max_size is None:
#             max_size=min_size
#         self.max_size=max_size
    
#     def __call__(self,image,target):
#         size=random.randint(self.min_size,self.max_size)
#         image=F.resize(image,size)
#         target=F.resize(target,size,interpolation=Image.NEAREST)
#         return image,target

# 固定resize
class Resize(object):
    def __init__(self,output_size=384,train=True) -> None:
        self.size=output_size
        self.train=train

    def __call__(self, image,target, instance_masks=None):
        image = F.resize(image, (self.size,self.size))
        # we must need to test on the original size 
        if self.train:
            target = F.resize(target, (self.size, self.size), interpolation=InterpolationMode.NEAREST)
        if instance_masks is not None:
            # instance_masks['masks'] = cv2.resize(instance_masks['masks'], 
            #                                     (self.size, self.size), 
            #                                     interpolation=cv2.INTER_NEAREST)
            return image, target, instance_masks     
        return image, target




class ToTensor(object):
    def __call__(self, image, target, instance_masks=None):
        image = F.to_tensor(image)
        target = torch.tensor(np.asarray(target), dtype=torch.int64)
        if instance_masks is not None:
            # instance_masks['masks'] = torch.tensor(np.asarray(instance_masks['masks']), dtype=torch.int64)
            return image, target, instance_masks
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, instance_masks=None):
        # print(image.shape, '.,..')
        image = F.normalize(image, mean=self.mean, std=self.std)
        if instance_masks is not None:
            return image, target, instance_masks 
        return image, target


# We don't apply other complex data augment
def get_transform(size, train=True):
    transforms = []
    transforms.append(Resize(size, train))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return Compose(transforms)

