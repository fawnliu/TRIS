import os
# os.environ['CUDA_ENABLE_DEVICES'] = '2,3'

import torch 
import os
import torch.nn.functional as F

from model.model_stage1 import TRIS 
# from model.model_stage2 import TRIS 

import torch.distributed as dist
from dataset.Dataset_referit import get_refit_dataset
from args import get_parser
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.util import AverageMeter, load_checkpoint
import time 
import datetime
import numpy  as np 
import cv2 
from utils.util import compute_mask_IU 
import torch.nn as nn 
from tensorboardX import SummaryWriter
from utils.box_eval_utils import generate_bbox,eval_box_iou, eval_box_acc

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
    _, val_dataset = get_refit_dataset(args)  

    val_datasets = []
    for test_split in args.test_split.split(','):
        val_datasets.append(val_dataset)

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
            load_checkpoint(args, model_without_ddp)  #####
        if args.eval:
            st = time.time()
            val_acc, testA_acc, testB_acc = 0, 0, 0
            for i, val_loader in enumerate(val_loaders):
                oIoU, mIoU, hit = validate(args, val_loader, model, local_rank)
                if i == 0: val_acc = mIoU
                elif i == 1: testA_acc = mIoU
                else: testB_acc = mIoU
            print(f'val: {val_acc}, testA, {testA_acc}, testB: {testB_acc}')
            all_t = time.time() - st 
            print(f'Testing time:  {str(datetime.timedelta(seconds=int(all_t)))}')
            return

def isCorrectHit(bbox_annot, heatmap, gt_mask=None):
    # H, W = orig_img_shape
    # heatmap_resized = cv2.resize(heatmap, (W, H))
    max_loc = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)

    ## get point mask 
    hitm = 0 
    new_gt_mask = gt_mask[max_loc[0], max_loc[1]] + 1
    if new_gt_mask.max() == 2:
        hitm = 1 

    for bbox in bbox_annot:
        if bbox[0] <= max_loc[1] <= bbox[2] and bbox[1] <= max_loc[0] <= bbox[3]:
            return 1, max_loc, hitm
    return 0, max_loc, hitm 


@torch.no_grad()
def validate(args,data_loader,model,local_rank=0, visualize=False):
    num_steps = len(data_loader)
    model.eval()

    batch_time=AverageMeter()
    mIOU_meter=AverageMeter()
    # I_meter=AverageMeter()
    # U_meter=AverageMeter() 
    box_mIOU_meter = AverageMeter()
    box_Acc_meter = AverageMeter()

    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    seg_total = 0.
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)

    end=time.time()
    start = time.time()
    len_data_loader = 0 
    hit_acc = 0 
    hitmask_acc = 0 
    for idx,(img, samples, image_sizes, img_path) in enumerate(data_loader):
        img_id = img_path[0].split('/')[-1].split('.')[0]
        img = img.cuda()

        j = 0 
        for sen in samples.keys():
            len_data_loader += 1

            item = samples[sen]
            sentences, bbox = item['sentences'], item['bbox']
            bbox = bbox[0]
            word_id = item['word_id'].cuda() 
            target = item['mask'].cuda() 
            o_H,o_W = target.shape[-2:]
            batch_size = word_id.shape[0]

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
            hit, max_loc, hitmask = isCorrectHit(bbox.numpy(), t_cam.cpu().numpy().astype(np.float64), target)
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
        
            # I, U = compute_mask_IU(target, pred)
            IoU=I*1.0/U # [overall IOU of batch]
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (I / U >= eval_seg_iou)
            seg_total += 1

            mIOU_meter.update(IoU,batch_size)
            box_mIOU_meter.update(box_miou, batch_size)
            box_Acc_meter.update(box_accu, batch_size)

            batch_time.update(time.time()-end)
            end=time.time()
            
            if idx % args.print_freq==0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas=batch_time.avg*(num_steps-idx)
                print(
                    f'Test: [{idx:5d}/{len(data_loader)}] | '
                    f'mIOU {100*mIOU_meter.avg:.3f} | '
                    f'Overall IOU {100*float(cum_I)/float(cum_U):.3f} | '
                    f'Hit {hit_acc/len_data_loader*100:.3f} | '
                    f'HitM {hitmask_acc/len_data_loader*100:.3f} | '
                    f'box_mIOU {100*box_mIOU_meter.avg:.3f} | '
                    f'box_Acc {100*box_Acc_meter.avg:.3f} | '
                    f'eta: {datetime.timedelta(seconds=int(etas))} | '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})', flush=True)

    overall_IoU = 100*float(cum_I)/float(cum_U)
    mIOU = 100*mIOU_meter.avg
    hit = 100*hit_acc/len_data_loader 
    box_miou = 100*box_mIOU_meter.avg
    box_acc = 100*box_Acc_meter.avg
    print(f'Test: mIOU {mIOU:.5f}  \
            Overall IOU {overall_IoU:.5f}  \
            HiT {100*hit_acc/len_data_loader:.3f}  \
            hitmax_acc {hitmask_acc/len_data_loader*100:.3f} \
            box_mIOU {box_miou.data.cpu().numpy()} \
            box_acc {box_acc.data}')
    
    return overall_IoU, mIOU, hit 




if __name__=="__main__":
    parse=get_parser()
    args=parse.parse_args()

    print('========='*10)
    print(args)
    print('========='*10)

    if args.vis_out is not None and not os.path.exists(args.vis_out):
        os.mkdir(args.vis_out)

    if args.cam_save_dir is not None and not os.path.exists(args.cam_save_dir ):
        os.mkdir(os.path.join(args.cam_save_dir))
    
    if args.eval_vis_out is not None and not os.path.exists(args.eval_vis_out) and args.visualize:
        os.makedirs(args.eval_vis_out)

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
    
    
    global writer 
    if args.board_folder is not None:
        writer = SummaryWriter(args.board_folder)

    main(args) 


