import pickle
import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import os, pickle, cv2
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

import CLIP.clip as clip 
import scipy.io as sio

from pycocotools import mask as cocomask

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_referit_gt_mask(mask_path):
    mat = sio.loadmat(mask_path)
    mask = (mat['segimg_t'] == 0)
    return mask

def save_tmp_mask(input_path, save_name):
    m1 = load_referit_gt_mask(input_path)
    cv2.imwrite(save_name, m1*255)

import torchvision.transforms as transforms
def get_flicker_transform(args):
    Isize = args.size 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    resize = (Isize, Isize)
    tflist = [transforms.Resize(resize)]

    transform_train = transforms.Compose(tflist + [
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])

    transform_test = transforms.Compose([
                             transforms.Resize(resize),
                             transforms.ToTensor(),
                             normalize
                             ])

    return transform_train, transform_test


class ImageLoader_train(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", loader=pil_loader, max_tokens=20, negative_samples=0, pseudo_path=None):
        annt_path = os.path.join(root, 'annotations', split + '.pickle')
        print(annt_path, '----', 'Train')
        with open(annt_path, 'rb') as f:
            self.annotations = pickle.load(f, encoding='latin1')
        self.files = list(self.annotations.keys())
        print('num of data:{}'.format(len(self.files)))
        self.transform = transform
        self.loader = loader
        self.split = split
        self.img_folder = os.path.join(root, 'images')

        self.max_tokens = max_tokens 
        self.negative_samples = negative_samples

        self.pseudo_path = pseudo_path

        self.all_refs = {} 
        # self.imgid2_refs = {} 
        # self.refid2index = {} 
        index_g = 0
        for index in range(len((self.files))):
            item = str(self.files[index])
            ann = self.annotations[item]['annotations']

            # self.imgid2_refs[ann[0]['image_id']] = ann 

            for ref in ann:
                self.all_refs[index_g] = ref 
                # self.refid2index[ref['ref_id']] = index_g

                index_g += 1 
        print(index_g, '===')

    def __getitem__(self, index):
        item = self.all_refs[index]
        
        img_path = os.path.join(self.img_folder, str(item['image_id']) + '.jpg')
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)   

        query = item['query']
        word_id = clip.tokenize(query).squeeze(0)[:self.max_tokens]
        word_id = np.array(word_id)

        return img, word_id, -1

    def __len__(self):
        return len(self.all_refs) * 1


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, split="train", loader=pil_loader, max_tokens=20):
        annt_path = os.path.join(root, 'annotations', split + '.pickle')
        print(annt_path, '----')
        with open(annt_path, 'rb') as f:
            self.annotations = pickle.load(f, encoding='latin1')
        self.files = list(self.annotations.keys())
        print('num of data:{}'.format(len(self.files)))
        self.transform = transform
        self.loader = loader
        self.split = split
        self.img_folder = os.path.join(root, 'images')

        self.max_tokens = max_tokens 

    def __getitem__(self, index):
        item = str(self.files[index])
        img_path = os.path.join(self.img_folder, item + '.jpg')
        img = Image.open(img_path).convert("RGB")
        image_sizes = (img.height, img.width)

        img = self.transform(img)

        ann = self.annotations[item]['annotations']
        
        out = {}
        for i in range(0, len(ann)):
            tmp = {}
            bbox = ann[i]['bbox']

            if (bbox[0][3]-bbox[0][1]) * (bbox[0][2]-bbox[0][0]) > 0.05 * image_sizes[0] * image_sizes[1]:
                tmp['sentences'] = ann[i]['query']
                tmp['word_id'] = clip.tokenize(ann[i]['query']).squeeze(0)[:self.max_tokens]
                tmp['bbox'] = np.array(bbox)
                ####### get target mask 
                rle = ann[i]['segmentation']
                mask = cocomask.decode(rle)
                mask = np.sum(mask, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                mask = mask.astype(np.uint8) # convert to np.uint8
                # img, mask = self.image_transforms(img, mask) 
                #######
                tmp['mask'] = mask 
                out[str(i)] = tmp 
        return img, out, image_sizes, img_path 

    def __len__(self):
        return len(self.files) 




from dataset.transform import get_transform
def get_refit_dataset(args):
    datadir = args.refer_data_root
    transform_train, transform_test = get_flicker_transform(args)
    ds_train = ImageLoader_train(datadir, split='train', transform=transform_train, max_tokens=args.max_query_len, negative_samples=args.negative_samples)  
    ds_test = ImageLoader(datadir, split='test', transform=transform_test, max_tokens=args.max_query_len)
    return ds_train, ds_test

if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-Isize', '--Isize', default=224, help='image size', required=False)
    args = vars(parser.parse_args())
    ds = get_refit_dataset(args=args)
    ds = torch.utils.data.DataLoader(ds,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=False,
                                     drop_last=False)
    pbar = tqdm(ds)
    # for i, (real_imgs, text) in enumerate(pbar):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (real_imgs, meta, size) in enumerate(pbar):
        size = [int(size[1]), int(size[0])]
        for sen in meta.keys():
            image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            image = (image - image.min()) / (image.max() - image.min())
            image = np.array(255 * image).copy().astype(np.uint8)
            image = cv2.resize(image, size)
            item = meta[sen]
            text, bbox = item['sentences'], item['bbox']
            bbox = torch.tensor(bbox)

            x1, x2, y1, y2 = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            bbox_norm = [x1 / width, y1 / height, x2 / width, y2 / height]
            if max(bbox_norm) > 1:
                continue
        #     (gxa, gya, gxb, gyb) = list(bbox.squeeze())
        #     image = cv2.rectangle(image, (int(gxa), int(gya)), (int(gxb), int(gyb)), (0, 0, 255), 2)
        #     cv2.putText(image, text[0], (int(gxa)+10, int(gya)), font, fontScale=0.3,
        #                 color=(0, 0, 0),
        #                 thickness=1)
        #     cv2.imwrite('kaki.jpg', image)
        #     pass
        # pass

