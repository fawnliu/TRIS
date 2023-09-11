import os
import torch 
import os
import torch.nn.functional as F

from model.model_stage1 import TRIS 
# from model.model_stage2 import TRIS 

import torch.distributed as dist

from dataset.ReferDataset import ReferDataset 

from dataset.transform import get_transform
from args import get_parser
# import config 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.util import AverageMeter, load_checkpoint
import time 
from logger import create_logger
import datetime
import numpy  as np 
import cv2 
from utils.util import compute_mask_IU 
import torch.nn as nn 
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from utils.box_eval_utils import eval_box_iou, generate_bbox, eval_box_acc

# --------------------- set random seed -------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1234)
# -------------------------------------------------------------------------


def main(args):
    if args.distributed:
        local_rank=dist.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    # build module
    model = TRIS(args)
    if args.distributed:
        model.cuda(local_rank)
    else:
        model.cuda() 

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
    else:
        model=torch.nn.DataParallel(model)

    model_without_ddp=model.module
    num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {num_params}")
    # build dataset
    val_datasets = [] 
    for test_split in args.test_split.split(','):
        val_datasets.append(ReferDataset(refer_data_root=args.refer_data_root,
                                            dataset=args.dataset,
                                            splitBy=args.splitBy,
                                            bert_tokenizer=args.bert_tokenizer,
                                            split=test_split,
                                            size=args.size,
                                            image_transforms=get_transform(args.size, train=False),
                                            eval_mode=True,
                                            scales=args.scales,
                                            max_tokens=args.max_query_len,
                                            positive_samples=args.positive_samples,
                                            pseudo_path=args.pseudo_path))  ######## 1 for multitext inference, else with same with train datasets
    if args.distributed:
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(DistributedSampler(val_dataset, shuffle=False))
    else:
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(None)

    val_loaders = [] 
    for val_dataset, val_sampler in zip(val_datasets, val_samplers):
        val_loaders.append(DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=2,
                            pin_memory=True, 
                            sampler=val_sampler,
                            shuffle=False))
    
    if args.resume:
        if args.pretrain is not None:
            load_checkpoint(args, model_without_ddp, logger=logger)  #####
        if args.eval:
            # validate(args,val_loader,model,local_rank)
            st = time.time()
            val_acc, testA_acc, testB_acc = 0, 0, 0
            for i, val_loader in enumerate(val_loaders):
                if args.prms:
                    oIoU, mIoU, hit = validate_same_sentence(args, val_loader, model, local_rank)
                else:
                    oIoU, mIoU, hit = validate(args, val_loader, model, local_rank)
                if i == 0: val_acc = mIoU
                elif i == 1: testA_acc = mIoU
                else: testB_acc = mIoU
                print()
                print()
            print(f'val: {val_acc}, testA, {testA_acc}, testB: {testB_acc}')
            all_t = time.time() - st 
            print(f'Testing time:  {str(datetime.timedelta(seconds=int(all_t)))}')
            # return

    # # ########
    # oIoU1, mIoU1, hit1 = validate_same_sentence(args, val_loader, model, local_rank)
    # print('same sents: ', oIoU1, mIoU1, hit1)
    # print(oIoU, mIoU, hit)
    # # ########


import json 

def isCorrectHit(bbox_annot, heatmap, gt_mask=None):
    max_loc = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)

    hitm = 0 
    max_point_score = gt_mask[max_loc[0], max_loc[1]] + 1
    if max_point_score.max() == 2:
        hitm = 1 

    for bbox in bbox_annot:
        if bbox[0] <= max_loc[1] <= bbox[2] and bbox[1] <= max_loc[0] <= bbox[3]:
            return 1, max_loc, hitm
    return 0, max_loc, hitm 

import CLIP.clip as clip 

def get_scores(clip_model, fg_224_eval, word_id):
    image_features = clip_model.encode_image(fg_224_eval)  # [N1, C]
    _, text_features = clip_model.encode_text(word_id)  # [N2, C]
    # normalization
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits_per_image = (image_features @ text_features.t())  # [N1, N2]
    return logits_per_image 



@torch.no_grad()
def validate(args, data_loader, model, local_rank=0, visualize=False, logger=None, save_cam=False):
    num_steps = len(data_loader)
    model.eval()

    if save_cam:
        os.makedirs(args.name_save_dir, exist_ok=True)
    if save_cam:
        os.makedirs(args.cam_save_dir, exist_ok=True)
        
    batch_time=AverageMeter()
    mIOU_meter=AverageMeter()
    I_meter=AverageMeter()
    U_meter=AverageMeter() 
    box_mIOU_meter = AverageMeter()
    box_Acc_meter = AverageMeter()

    start = time.time()
    end=time.time()
    len_data_loader = 0 
    hit_acc = 0 
    hitmask_acc = 0  
    cam_out_name = [] 
    for idx,(samples, targets) in enumerate(data_loader):
        img_id = targets['img_path'].numpy()[0]
        
        word_ids = samples['word_ids'].squeeze(1)
        word_masks = samples['word_masks'].squeeze(1)
        # ms_img_list = samples['ms_img_list']#.cuda(local_rank, non_blocking=True)
        img = samples['img'].cuda(local_rank,non_blocking=True) # [B,3,H,W]
        batch_size = img.size(0)
        target = targets['target'].cuda(local_rank,non_blocking=True) #[B,ori_H,ori_W]
        word_ids = word_ids.cuda(local_rank,non_blocking=True) # [B,len] or [B,len,num]
        word_masks = word_masks.cuda(local_rank,non_blocking=True) # [B,len] or [B,len,num]
        bbox = targets['boxes']#.cuda(local_rank,non_blocking=True) 
        sentences = targets['sentences']

        o_H,o_W = target.shape[-2:]

        for j in range(word_ids.size(-1)):
            len_data_loader += 1
            o_H, o_W = target.shape[-2:]
            word_id = word_ids[:,:,j]
            word_mask = word_masks[:,:,j]

            output = model(img, word_id)  
            pred = F.interpolate(output, (o_H,o_W), align_corners=True, mode='bilinear').squeeze(0)

            # pdb.set_trace() 
            pred /= F.adaptive_max_pool2d(pred, (1, 1)) + 1e-5
            pred = pred.squeeze(0)
            t_cam = pred.clone()
            pred = pred.gt(1e-9)
            target = target.squeeze(0).squeeze(0)     
            
            I, U = compute_mask_IU(target, pred)
            IoU = I*1.0/U 
            hit, max_loc, hitmask = isCorrectHit(bbox.numpy(), t_cam.cpu().numpy().astype(np.float32), target)
            hit_acc += hit ########
            hitmask_acc += hitmask 
            #######
            bbox_gen = generate_bbox(pred.cpu().numpy().astype(np.float64))
            bbox_hit = bbox_gen[0]
            for bb in bbox_gen:
                if bb[0] <= max_loc[1] <= bb[2] and bb[1] <= max_loc[0] <= bb[3]:
                    bbox_hit = bb 
            box_miou = eval_box_iou(torch.tensor(bbox_hit[0:4]).unsqueeze(0), bbox)
            box_accu = eval_box_acc(bbox_gen, bbox) ### !!!box_acc for all generated boxes 
            #######

            I_meter.update(I)
            U_meter.update(U)
            mIOU_meter.update(IoU, batch_size)
            box_mIOU_meter.update(box_miou, batch_size)
            box_Acc_meter.update(box_accu, batch_size)

            if args.cam_save_dir is not None and save_cam:
                root = os.path.join(args.cam_save_dir, f'{idx}_{j}_{img_id}.npy')
                np.save(root, t_cam.cpu().numpy())
            if args.name_save_dir is not None and save_cam:
                cam_out_name.append(f'{idx}_{j}_{img_id}')

        batch_time.update(time.time()-end)
        end=time.time()

        if idx % args.print_freq==0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas=batch_time.avg*(num_steps-idx)
            print(
                f'Test: [{idx:4d}/{num_steps}] | '
                f'mIOU {100*mIOU_meter.avg:.3f} | '
                f'Overall IOU {100*float(I_meter.sum)/float(U_meter.sum):.3f} | '
                f'Hit {hit_acc/len_data_loader*100:.3f} | '
                f'HitM {hitmask_acc/len_data_loader*100:.3f} | '
                f'box_mIOU {100*box_mIOU_meter.avg:.3f} | '
                f'box_Acc {100*box_Acc_meter.avg:.3f} | '
                f'eta: {datetime.timedelta(seconds=int(etas))} || '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})', flush=True)
    
    if args.name_save_dir is not None and save_cam:
        with open(os.path.join(args.name_save_dir, f'{args.dataset}_train_cam_name.json'), 'w') as f:
            f.write(json.dumps(cam_out_name))

    overall_IoU = 100*float(I_meter.sum)/float(U_meter.sum)
    mIOU = 100*mIOU_meter.avg
    hit = 100*hit_acc/len_data_loader 
    box_miou = 100*box_mIOU_meter.avg
    box_acc = 100*box_Acc_meter.avg
    print(f'Test: mIOU {mIOU:.5f}  \
            Overall IOU {overall_IoU:.5f}  \
            HiT {100*hit_acc/len_data_loader:.3f}  \
            HitM {hitmask_acc/len_data_loader*100:.3f} \
            box_mIOU {box_miou.data.cpu().numpy()} \
            box_acc {box_acc.data}')
    
    return overall_IoU, mIOU, hit 


@torch.no_grad()
def validate_same_sentence(args, data_loader, model, local_rank=0, visualize=False, logger=None, save_cam=False):
    num_steps = len(data_loader)
    model.eval()

    save_cam = args.save_cam 

    if save_cam and not os.path.exists(args.name_save_dir):
        os.makedirs(args.name_save_dir, exist_ok=True)
    if save_cam and not os.path.exists(args.cam_save_dir):
        os.makedirs(args.cam_save_dir, exist_ok=True)

    batch_time=AverageMeter()
    mIOU_meter=AverageMeter()
    I_meter=AverageMeter()
    U_meter=AverageMeter() 

    start = time.time()
    end=time.time()
    len_data_loader = 0 
    hit_acc = 0 
    hitmask_acc = 0

    clip_input_size = 224 
    ###############
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False, txt_length=args.max_query_len)
    clip_model.eval()
    ###############
    cam_out_name = [] 
    for idx,(samples, targets) in enumerate(data_loader):
        # if (idx+1) % 100 == 0:
        #     break 

        img_id = targets['img_path'].numpy()[0]
        
        word_ids = samples['word_ids'].squeeze(1)
        word_masks = samples['word_masks'].squeeze(1)
        img = samples['img'].cuda(local_rank,non_blocking=True) # [B,3,H,W]
        batch_size = img.size(0)
        target = targets['target'].cuda(local_rank,non_blocking=True) #[B,ori_H,ori_W]
        word_ids = word_ids.cuda(local_rank,non_blocking=True) # [B,len] or [B,len,num]
        word_masks = word_masks.cuda(local_rank,non_blocking=True) # [B,len] or [B,len,num]
        bbox = targets['boxes'].cuda(local_rank,non_blocking=True) 
        sentences = targets['sentences']

        o_H,o_W = target.shape[-2:]

        img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
        max_info = {
            'score': -1,
            'index': -1,
            'cam': 0
        }

        for j in range(word_ids.size(-1)):
            len_data_loader += 1
            o_H, o_W = target.shape[-2:]
            word_id = word_ids[:,:,j]
            word_mask = word_masks[:,:,j]
            
            output = model(img, word_id)
            pred = F.interpolate(output, (o_H,o_W), align_corners=True, mode='bilinear').squeeze(0)

            cam_224 = F.interpolate(output, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
            fg_224_eval = []
            for i in range(len(img_224)):
                fg_224_eval.append(cam_224[i] * img_224[i])
            fg_224_eval = torch.stack(fg_224_eval, dim=0)            

            score = 0.
            for _i in range(word_ids.size(-1)):
                score += get_scores(clip_model, fg_224_eval, word_ids[:,:,_i]).item() 
            if score > max_info['score']:
                max_info['score'] = score
                max_info['index'] = j 
                max_info['cam'] = pred 
        
        pred = max_info['cam']

        pred /= F.adaptive_max_pool2d(pred, (1, 1)) + 1e-5
        pred = pred.squeeze(0)
        t_cam = pred.clone()
        pred = pred.gt(1e-9)
        target = target.squeeze(0).squeeze(0)     
        
        I, U = compute_mask_IU(target, pred)
        I = I*word_ids.size(-1)
        U = U*word_ids.size(-1)
        IoU = I*1.0/U 
        hit, max_loc, hitmask = isCorrectHit(bbox.cpu().numpy(), t_cam.cpu().numpy().astype(np.float32), target)
        hit_acc += hit * word_ids.size(-1) ########
        hitmask_acc += hitmask * word_ids.size(-1)

        I_meter.update(I, batch_size*word_ids.size(-1))
        U_meter.update(U, batch_size*word_ids.size(-1))
        mIOU_meter.update(IoU, batch_size*word_ids.size(-1))

        if args.cam_save_dir is not None and save_cam:
            root = os.path.join(args.cam_save_dir, f'{idx}_{img_id}.npy')
            # root = os.path.join(args.cam_save_dir, 'cam', f'{idx}_{img_id}.npy')
            np.save(root, t_cam.cpu().numpy())
        if args.name_save_dir is not None and save_cam:
            cam_out_name.append(f'{idx}_{img_id}')

        batch_time.update(time.time()-end)
        end=time.time()

        if idx % args.print_freq==0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas=batch_time.avg*(num_steps-idx)
            print(
                f'Test: [{idx:4d}/{num_steps}] | '
                f'mIOU {100*mIOU_meter.avg:.3f} | '
                f'Overall IOU {100*float(I_meter.sum)/float(U_meter.sum):.3f} | '
                f'Hit {hit_acc/len_data_loader*100:.3f} | '
                f'HitM {hitmask_acc/len_data_loader*100:.3f} | '
                f'eta: {datetime.timedelta(seconds=int(etas))} || '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})', flush=True)
    
    if args.name_save_dir is not None and save_cam:
        with open(os.path.join(args.name_save_dir, f'{args.dataset}_train_names.json'), 'w') as f:
            f.write(json.dumps(cam_out_name))

    overall_IoU = 100*float(I_meter.sum)/float(U_meter.sum)
    mIOU = 100*mIOU_meter.avg
    hit = 100*hit_acc/len_data_loader 
    print(f'Test: mIOU {mIOU:.5f}  \
            Overall IOU {overall_IoU:.5f}  \
            HiT {100*hit_acc/len_data_loader:.3f}  \
            HitM {hitmask_acc/len_data_loader*100:.3f}')
    return overall_IoU, mIOU, hit 




if __name__=="__main__":
    parse=get_parser()
    args=parse.parse_args()

    print('========='*10)
    print(args)
    print('========='*10)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank=int(os.environ['RANK'])
        world_size=int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank=-1
        world_size=-1

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
    
    
    if args.distributed:
        logger = create_logger(dist_rank=dist.get_rank())
    else:
        logger = create_logger()
    
    global writer 
    if args.board_folder is not None:
        writer = SummaryWriter(args.board_folder)

    main(args) 


