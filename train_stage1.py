import os
# os.environ['CUDA_ENABLE_DEVICES'] = '2,3'
import torch 
import torch.nn.functional as F

from model.model_stage1 import TRIS

import torch.distributed as dist
from torch.optim import AdamW
from dataset.ReferDataset import ReferDataset 
from validate import validate 
from dataset.transform import get_transform
from args import get_parser
# import config 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.poly_lr_decay import PolynomialLRDecay
from utils.util import AverageMeter, load_checkpoint,reduce_tensor, save_checkpoint, load_pretrained_checkpoint
import time 
from logger import create_logger
import datetime
import random 
import numpy  as np 
import cv2 
from utils.util import compute_mask_IU 
import torch.nn as nn 
from tensorboardX import SummaryWriter
import CLIP.clip as clip 
import torchvision 

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
    try:
        param_groups = model.trainable_parameters()
    except:
        print() 
        param_groups = None 
        print('no param goups...')
        print() 

    if args.distributed:
        model.cuda(local_rank)
    else:
        model.cuda() 
    
    # #################
    # for param in model.backbone.parameters():
    #     param.require_grad = False 
    # #################

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
    else:
        model=torch.nn.DataParallel(model)

    model_without_ddp=model.module
    print() 
    print(model_without_ddp)
    print()
    # model.train() 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params / 1e6: .2f}M")
    # build dataset
    train_dataset = ReferDataset(refer_data_root=args.refer_data_root,
                                dataset=args.dataset,
                                splitBy=args.splitBy,
                                bert_tokenizer=args.bert_tokenizer,
                                split='train',
                                size=args.size,
                                max_tokens=args.max_query_len,
                                image_transforms=get_transform(args.size, train=True),
                                eval_mode=args.eval,
                                negative_samples=args.negative_samples,
                                positive_samples=args.positive_samples)
    val_datasets = [] 
    for test_split in args.test_split.split(','):
        val_datasets.append(ReferDataset(refer_data_root=args.refer_data_root,
                                            dataset=args.dataset,
                                            splitBy=args.splitBy,
                                            bert_tokenizer=args.bert_tokenizer,
                                            split=test_split,
                                            size=args.size,
                                            max_tokens=args.max_query_len,
                                            image_transforms=get_transform(args.size, train=False),
                                            eval_mode=True,
                                            scales=args.scales,
                                            positive_samples=args.positive_samples)) 
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(DistributedSampler(val_dataset, shuffle=False))
    else:
        train_sampler = None 
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(None)

    train_loader=DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=2,
                            pin_memory=True,
                            sampler=train_sampler,
                            shuffle=(train_sampler is None))
    val_loaders = [] 
    for val_dataset, val_sampler in zip(val_datasets, val_samplers):
        val_loaders.append(DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=2,
                            pin_memory=True, 
                            sampler=val_sampler,
                            shuffle=False))

    if param_groups is not None:
        optimizer = AdamW([
            {'params': param_groups[0], 'lr': args.lr * args.lr_multi, 'weight_decay': args.weight_decay},
            {'params': param_groups[1], 'lr': args.lr, 'weight_decay': args.weight_decay},
        ], lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('param_groups is None !')
        optimizer = AdamW(params=model.parameters(), 
                      lr=args.lr, 
                      weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda x: (1 - x / (len(train_loader) * args.epoch)) ** 0.9)

    print()
    print(optimizer)
    print(scheduler)
    print()
    
    if args.resume:
        if args.pretrain is not None:
            load_checkpoint(args, model_without_ddp, optimizer, scheduler, logger)  #####
        if args.eval:
            # validate(args,val_loader,model,local_rank)
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
    
    # load pretrained model from vg 
    if args.pretrained_checkpoint is not None:
        print('loading ', args.pretrained_checkpoint)
        load_pretrained_checkpoint(args.pretrained_checkpoint, model_without_ddp)

    logger.info("Start training")

    ###############
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp_max_len = args.max_query_len
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False, txt_length=tmp_max_len)
    clip_model.eval()
    ###############

    train_time = 0
    start_time = time.time()
    best = {
        'val_acc': -1,
        'val_hit': -1,
        'epoch': -1,
        'path': '',
        'hit': -1,
        'hit_path': '',
        'testA': -1,
        'testB': -1 
    }
    iteration = 0
    for epoch in range(args.start_epoch, args.epoch):
        st = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        iteration = train_one_epoch(train_loader, model, optimizer, epoch, local_rank, args, iteration, clip_model, lr_scheduler=scheduler)

        train_time += time.time() - st 

        # validation, save best 
        val_acc, testA_acc, testB_acc = 0, 0, 0
        for i, val_loader in enumerate(val_loaders):
            oIoU, mIoU, hit = validate(args, val_loader, model, local_rank)
            if i == 0: val_acc = mIoU 
            elif i == 1: testA_acc = mIoU 
            else: testB_acc = mIoU 
        # save_checkpoint(epoch,model_without_ddp,optimizer,scheduler,logger,args, f'ckpt_320_epoch_{epoch}.pth') #### 
        if val_acc > best['val_acc'] and local_rank==0:
            if os.path.exists(best['path']):
                print('remove ', best['path'])
                os.remove(best['path'])
            save_path = save_checkpoint(epoch,model_without_ddp,optimizer,scheduler,logger,args, f'ckpt_320_epoch_{epoch}_best.pth')
            best['val_acc'] = val_acc.item() 
            best['val_hit'] = hit 
            best['epoch'] = epoch 
            best['path'] = save_path
            best['testA'] = testA_acc
            best['testB'] = testB_acc
        if hit > best['hit'] and local_rank==0:
            best['hit'] = hit 
            if os.path.exists(best['hit_path']):
                print('remove ', best['hit_path'])
                os.remove(best['hit_path'])
            save_path = save_checkpoint(epoch,model_without_ddp,optimizer,scheduler,logger,args, f'ckpt_320_epoch_{epoch}_hit.pth')
            best['hit_path'] = save_path
            best['hit'] = hit 
        print(best)

        if local_rank == 0 and args.board_folder is not None:
            writer.add_scalar('test/mIoU', val_acc, epoch)
            writer.add_scalar('test/hit', hit, epoch)
            writer.add_scalar('test/oIoU', oIoU, epoch)
    
    # ############ validate on the train dataset
    # print()
    # print()
    # last_trainset = ReferDataset(refer_data_root=args.refer_data_root,
    #                         dataset=args.dataset,
    #                         split='train',
    #                         splitBy=args.splitBy,
    #                         image_transforms=get_transform(args.size, train=False),
    #                         eval_mode=True,
    #                         size=args.size,
    #                         bert_tokenizer=args.bert_tokenizer)
    # val_train_loader = DataLoader(last_trainset,
    #                         batch_size=1,
    #                         num_workers=2,
    #                         pin_memory=True, 
    #                         sampler=val_sampler)
    # print('loading ', best['path'])
    # load_pretrained_checkpoint(best['path'], model_without_ddp)
    # oIoU_1, mIoU_1, hit_1 = validate(args, val_train_loader, model, local_rank)
    # print('Validat on the train split: ', oIoU_1, mIoU_1, hit_1)
    # print(best)
    # # # ############ validate on the train dataset, with prms strategy 
    # print()
    # print()
    # print('--------same sents--------')
    # print()
    # from validate import validate_same_sentence 
    # oIoU, mIoU, hit = validate_same_sentence(args, val_train_loader, model, local_rank, save_cam=False)
    # print() 
    # print('Validat on the train split (same sents): ', oIoU, mIoU, hit)
    # print('Validat on the train split: ', oIoU_1, mIoU_1, hit_1)
    # print(best)
    # # # ##################

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(train_time))
    logger.info('Training + testing time {}'.format(total_time_str))


from loss.clip_loss import clip_forward


def train_one_epoch(train_loader,model,optimizer,epoch,local_rank,args, iteration=0, clip_model=None, lr_scheduler=None):
    num_steps=len(train_loader)
    model.train()

    batch_time=AverageMeter()
    loss_meter=AverageMeter()

    start=time.time()
    end=time.time()

    max_iter = int(num_steps * args.epoch) 

    clip_input_size = 224 

    for idx,(samples, targets) in enumerate(train_loader):
        word_ids = samples['word_ids'].squeeze(1)
        word_masks = samples['word_masks']#.squeeze(1)
        img = samples['img'].cuda(local_rank,non_blocking=True)
        target = targets['target'].cuda(local_rank,non_blocking=True)
        bbox = targets['boxes'].cuda(local_rank,non_blocking=True) 
        word_ids = word_ids.cuda(local_rank,non_blocking=True)
        word_masks = word_masks.cuda(local_rank,non_blocking=True)
        B,c,h,w = img.shape
        raw_sentences = targets['sentences']

        labels = torch.eye(B).cuda()  

        cls, _, seg_final_out, sig_out,  _ = model(img, word_ids)  

        if img.shape[2] != clip_input_size:
            cam_224 = F.interpolate(sig_out, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
            img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
        else:
            cam_224 = sig_out 
            img_224 = img 
        fg_224_eval = []
        bg_224_eval = [] 
        for i in range(len(img_224)):
            fg_224_eval.append(cam_224[i] * img_224[i])
            bg_224_eval.append((1 - cam_224[i]) * img_224[i])
        fg_224_eval = torch.stack(fg_224_eval, dim=0)
        bg_224_eval = torch.stack(bg_224_eval, dim=0)
        fg_loss = -(torch.log(clip_forward(clip_model, fg_224_eval, word_ids))).mean()
        
        if args.negative_samples > 0:
            neg_phrases = samples['neg_word_ids'].cuda()
            try:
                image_features = clip_model.encode_image(fg_224_eval)
                neg_loss = torch.tensor(.0, requires_grad=True, device='cuda:0') 
                for i_ in range(B):
                    _, text_features = clip_model.encode_text(neg_phrases[i_])
                    image_feature = image_features[i_].reshape(1, -1)
                    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    neg_score = torch.matmul(image_feature, text_features.transpose(0,1))
                    neg_loss = neg_loss + (-(torch.log(1 - neg_score)).mean())
                neg_loss /= B 
            except:
                import pdb 
                pdb.set_trace()

        cls_loss = F.multilabel_soft_margin_loss(cls, labels)
        
        l1 = fg_loss
        l2 = neg_loss  
        l3 = cls_loss  

        loss = l1 * args.w1 + l2 * args.w2 + l3 * args.w3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step() 

        torch.cuda.synchronize()

        if local_rank == 0 and args.board_folder is not None:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('optim/lr', lr, iteration)
            writer.add_scalar('train/loss', loss.data.cpu().numpy(), iteration)
            writer.add_scalar('train/l1', l1.data.cpu().numpy(), iteration)
            writer.add_scalar('train/l2', l2.data.cpu().numpy(), iteration)
            writer.add_scalar('train/l3', l3.data.cpu().numpy(), iteration)

        # measure time
        loss_meter.update(loss.item(), target.size(0))
        batch_time.update(time.time()-end)
        end=time.time()

        if idx % args.print_freq==0 and local_rank==0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas=batch_time.avg*(num_steps-idx)
            all_etas = batch_time.avg * (max_iter - iteration)
            logger.info(
                f'Train:[{epoch:2d}/{args.epoch}][{idx:4d}/{num_steps}] | '
                f'eta: {datetime.timedelta(seconds=int(etas))} | lr {lr:.6f} || '
                f'loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | '
                f'l1: {l1:.4f} | '
                f'l2: {l2:.4f} | '
                f'l3: {l3:.4f} | '
                f'time: {batch_time.val:.4f} ({batch_time.avg:.4f}) | '
                f'mem: {memory_used:.0f}MB || '
                f'all_eta: {datetime.timedelta(seconds=int(all_etas))}')
        iteration += 1
    epoch_time=time.time()-start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return iteration



if __name__=="__main__":
    parse=get_parser()
    args=parse.parse_args()

    print('========='*10)
    print(args)
    print('========='*10)

    if args.vis_out is not None and not os.path.exists(args.vis_out):
        os.mkdir(args.vis_out)

    print(f'[{args.w1}, {args.w2}, {args.w3}]')
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
    
    if args.board_folder is not None:
        global writer 
        writer = SummaryWriter(args.board_folder)

    main(args) 


