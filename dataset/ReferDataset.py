from inspect import getcallargs
import os 
import sys
from tkinter.tix import Tree
from traceback import print_tb
# from dataset.RefTR_Dataset import denorm 
import torch.utils.data as data
import torch 
import numpy as np 
from PIL import Image 
import cv2 
import transformers
from dataset.refer import REFER
import CLIP.clip as clip 
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import pdb 
import imageio

def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))

def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)

class ReferDataset(data.Dataset):
    def __init__(self,
                 refer_data_root='./data',
                 dataset='refcoco',
                 splitBy='unc',
                 bert_tokenizer='clip',
                 image_transforms=None,
                 max_tokens=20, 
                 split='train',
                 eval_mode=True,
                 size=448,
                 scales=False,
                 negative_samples=0,
                 positive_samples=1,
                 pseudo_path=None) -> None:
        """
        parameters:
            args: argparse obj
            image_transforms: transforms apply to image and mask
            max_tokens: determined the max length of token 
            split: ['train','val','testA','testB']
            eval_mode: whether in training or evaluating 
        """

        self.clip = ('clip' in bert_tokenizer)
        self.negative_samples = negative_samples
        self.positive_samples = positive_samples 
        self.classes=[]
        self.image_transforms=image_transforms
        self.split=split
        self.refer=REFER(refer_data_root, dataset, splitBy)
        self.scales = scales 
        self.size = size 
        self.pseudo_path = pseudo_path

        print('Preparing dataset .....')
        print(dataset, split)
        print(refer_data_root, dataset, splitBy) 
        print(f'pseudo_path = {pseudo_path}')

        self.max_tokens=max_tokens

        ref_ids=self.refer.getRefIds(split=self.split)
        img_ids=self.refer.getImgIds(ref_ids)
        # change dict to list
        all_imgs=self.refer.Imgs
        self.imgs=list(all_imgs[i] for i in img_ids)
        
        self.ref_ids=ref_ids
        self.tokenizer = clip.tokenize 
        
        self.eval_mode = eval_mode

        self.input_ids=[]
        self.word_masks=[]
        self.all_sentences = []
        # get negative samples, 
        self.refid2index = {}

        for index, r in enumerate(self.ref_ids):
            self.refid2index[r] = index 

            # for each image
            ref=self.refer.Refs[r]
            # List[Tensor] Tensor shape [1,len]
            sentences_for_ref=[]
            attentions_for_ref=[]
            sentence_raw_for_re = []

            # for each sentence
            for i,(el,sent_id) in enumerate(zip(ref['sentences'],ref['sent_ids'])):
                sentence_raw = el['sent']
                
                word_id = self.tokenizer(sentence_raw).squeeze(0)[:self.max_tokens]
                word_id = np.array(word_id)
                word_mask = np.array(word_id>0,dtype=int)

                sentences_for_ref.append(torch.tensor(word_id).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(word_mask).unsqueeze(0))
                sentence_raw_for_re.append(sentence_raw)

            self.input_ids.append(sentences_for_ref)
            self.word_masks.append(attentions_for_ref)
            self.all_sentences.append(sentence_raw_for_re)
        print('Dataset prepared!')

    def __len__(self):
        return len(self.ref_ids)
    
    def __getitem__(self,index):
        this_ref_id=self.ref_ids[index]
        this_img_id=self.refer.getImgIds(this_ref_id)
        this_img=self.refer.Imgs[this_img_id[0]]

        img_path = os.path.join(self.refer.IMAGE_DIR,this_img['file_name'])
        img=Image.open(img_path).convert("RGB")

        ref=self.refer.loadRefs(this_ref_id)[0]

        ## box format: x1y1x2y2
        bbox = self.refer.Anns[ref['ann_id']]['bbox']
        bbox = np.array(bbox, dtype=int)
        bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]

        ref_mask=np.array(self.refer.getMask(ref)['mask'])
        annot=np.zeros(ref_mask.shape)
        annot[ref_mask==1]=1 
        # convert it to a Pillow image
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")
        pseudo_gt = None 

        if self.image_transforms is not None:
            h, w = ref_mask.shape 
            # involves transform from PIL to tensor and mean and std normalization
            img, target = self.image_transforms(img, annot)
            # bbox[0], bbox[2] = bbox[0] * (self.size / w), bbox[2] * (self.size / w)
            # bbox[1], bbox[3] = bbox[1] * (self.size / h), bbox[3] * (self.size / h)
        else:
            target=annot
        
        if self.eval_mode:
            embedding=[]
            att=[]
            sentences = [] 

            for s in range(len(self.input_ids[index])):
                e=self.input_ids[index][s]
                a=self.word_masks[index][s]
                sent = self.all_sentences[index][s]

                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
                sentences.append(sent)
            # all sentence
            word_ids = torch.cat(embedding, dim=-1)
            word_masks = torch.cat(att, dim=-1)
        else: # for training, random select one sentence 
            choice_sent = np.random.choice(len(self.input_ids[index]))
            word_ids = self.input_ids[index][choice_sent]
            word_masks = self.word_masks[index][choice_sent]
            sentences = self.all_sentences[index][choice_sent]

            if self.pseudo_path is not None:
                # pseudo_filename = f'{index}_{choice_sent}_{this_img_id[0]}.npy'
                pseudo_filename = f'{index}_{this_img_id[0]}.npy'
                pseudo_info = np.load(os.path.join(self.pseudo_path, pseudo_filename), allow_pickle=True).item()
                pseudo_gt = pseudo_info['mask']*1.0 
                pseudo_gt = pseudo_gt.sum(0)
                pseudo_gt = F.resize(Image.fromarray(pseudo_gt), (self.size, self.size), interpolation=InterpolationMode.NEAREST)
                pseudo_gt = torch.tensor(np.asarray(pseudo_gt), dtype=torch.int64).unsqueeze(0)
                # pseudo_gt[pseudo_gt>0] = 1   
            else:
                pseudo_gt = None 

            if self.negative_samples > 0:
                ###########
                img2ref = self.refer.imgToRefs[this_img_id[0]]
                neg_index = []
                for item in img2ref:
                    t_ref_id = item['ref_id']
                    t_category_id = item['category_id']
                    try:
                        if t_ref_id != this_ref_id:  # and this_category_id == t_category_id
                            neg_index.append(self.refid2index[t_ref_id])
                    except: ### for refcocog google, its refindex is not match
                        break 
                        import pdb
                        pdb.set_trace() 
                ###########

                if len(neg_index) > 0:
                    neg_sents = []
                    neg_word_ids = []
                    ## random select negtive samples from same random index 
                    # n_index = neg_index[np.random.choice(len(neg_index))]
                    while len(neg_sents) < self.negative_samples:
                        ## different random index 
                        n_index = neg_index[np.random.choice(len(neg_index))]
                        choice_sent = np.random.choice(len(self.input_ids[n_index]))
                        neg_word_ids.append(self.input_ids[n_index][choice_sent])
                        neg_sents.append(self.all_sentences[n_index][choice_sent])
                    neg_word_ids = torch.cat(neg_word_ids, dim=0)
                else:
                    # random index, then randomly select one sentence 
                    neg_sents = []
                    neg_word_ids = []
                    while len(neg_sents) < self.negative_samples:
                        n_index = np.random.choice(len(self.input_ids))
                        choice_sent = np.random.choice(len(self.input_ids[n_index]))
                        tmp_sent = self.all_sentences[n_index][choice_sent]
                        if tmp_sent != sentences:
                            neg_sents.append(tmp_sent)
                            neg_word_ids.append(self.input_ids[n_index][choice_sent])
                    neg_word_ids = torch.cat(neg_word_ids, dim=0)

        img_path_full = this_img['file_name']
        img_path = int(img_path_full.split('.')[0].split('_')[-1])

        samples = {
            "img": img,
            "word_ids": word_ids,
            "word_masks": word_masks,
        }
        if self.negative_samples > 0:
            samples['neg_sents'] = neg_sents
            samples['neg_word_ids'] = neg_word_ids
        targets = {
            "target": target.unsqueeze(0),
            "img_path": img_path,
            "sentences": sentences,
            "boxes": bbox,
            "orig_size": np.array([h, w]),
            "img_path_full": img_path_full
        }
        if pseudo_gt is not None:
            targets['pseudo_gt'] = pseudo_gt
        return samples, targets

                
if __name__ == '__main__':
    from transform import get_transform
    import numpy as np 
    import json 
    from torch.utils.data import DataLoader

    refcoco_train = ReferDataset(dataset='refcoco', splitBy='unc', split='train', eval_mode=False, image_transforms=get_transform(320, train=False)) 
    train_loader=DataLoader(refcoco_train,
                            batch_size=12,
                            num_workers=2,
                            pin_memory=True,
                            sampler=None)
    for idx,(img, target, bbox, word_ids, word_mask, _, raw_sentences) in enumerate(train_loader):
        print(idx, img.shape)

        if idx > 10: break 
