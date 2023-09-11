import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch 
import os
import torch.nn.functional as F
from model.model_stage2 import TRIS, criterion 

import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from args import get_parser
import random
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, sampler
from utils.poly_lr_decay import PolynomialLRDecay
from utils.util import AverageMeter, load_checkpoint,reduce_tensor, save_checkpoint, load_pretrained_checkpoint
import time 
from logger import create_logger
import datetime
from utils.util import compute_mask_IU 
import cv2 
import numpy as np 
import pdb 
import CLIP.clip as clip 

from ema_pytorch import EMA
from validate import validate 
from tensorboardX import SummaryWriter

"""
Some infos about training LVAT
polynomial learning rate decay
"""
import torchvision

def main(args):
    if args.distributed:
        local_rank=dist.get_rank()
    else:
        local_rank = 0

    # build module
    model=TRIS(args)
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
    
    #######
    if args.model_ema:
        model_ema = EMA(model)
        print('ema initialized !')
    else:
        model_ema = None 
        print('no ema')
    ##  

    if args.distributed:
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
    else:
        model=torch.nn.DataParallel(model)

    model_without_ddp=model.module
    if local_rank == 0:
        print() 
        print(model_without_ddp)
        print()
    # because clip_model.eval() in the init function 
    model.train() ######
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params/1e6:.2f}M")
    
    # build dataset
    train_dataset=ReferDataset(refer_data_root=args.refer_data_root,
                            dataset=args.dataset,
                            split='train', 
                            splitBy=args.splitBy,
                            image_transforms=get_transform(args.size, train=True),
                            eval_mode=False, 
                            size=args.size,
                            bert_tokenizer=args.bert_tokenizer, 
                            pseudo_path=args.pseudo_path)
    val_datasets = []
    for test_split in args.test_split.split(','):
        val_datasets.append(ReferDataset(refer_data_root=args.refer_data_root,
                            dataset=args.dataset,
                            split=test_split,
                            splitBy=args.splitBy,
                            image_transforms=get_transform(args.size, train=False),
                            eval_mode=True,
                            size=args.size,
                            bert_tokenizer=args.bert_tokenizer))
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        # val_sampler = DistributedSampler(val_dataset)
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(DistributedSampler(val_dataset, shuffle=False))
    else:
        train_sampler = None 
        # val_sampler = None 
        val_samplers = []
        for val_dataset in val_datasets:
            val_samplers.append(None)
    train_loader=DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=2,
                            pin_memory=True,
                            sampler=train_sampler,
                            shuffle=(train_sampler is None)) ######drop_last=True
    val_loaders = []
    for val_dataset, val_sampler in zip(val_datasets, val_samplers):
        val_loaders.append(DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=2,
                            pin_memory=True, 
                            sampler=val_sampler))

    # build optimizer and lr scheduler
    if param_groups is not None:   # train clip-based model
        optimizer = AdamW([
                {'params': param_groups[0], 'lr': args.lr * args.lr_multi, 'weight_decay': args.weight_decay},
                {'params': param_groups[1], 'lr': args.lr, 'weight_decay': args.weight_decay},
            ], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                            lambda x: (1 - x / (len(train_loader) * args.epoch)) ** 0.9)
    else:   # train swin-based model 
        optimizer=AdamW(params=model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        scheduler=PolynomialLRDecay(optimizer,
                                    max_decay_steps=args.max_decay_steps,
                                    end_learning_rate=args.end_lr,
                                    power=args.power)
    print()
    print(optimizer)
    print(scheduler)
    print()

    if args.resume:
        load_checkpoint(args,model_without_ddp,optimizer,scheduler,logger)
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
    
    if args.pretrained_checkpoint is not None:
        print('loading ', args.pretrained_checkpoint)
        load_pretrained_checkpoint(args.pretrained_checkpoint, model_without_ddp)

    logger.info("Start training")

    ###############
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False, txt_length=args.max_query_len)
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
    for epoch in range(args.start_epoch,args.epoch):
        st = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        iteration = train_one_epoch(train_loader,model,optimizer,epoch,local_rank,args, iteration, clip_model, model_ema)
        scheduler.step()

        train_time += time.time() - st 

        # validation, save best 
        val_acc, testA_acc, testB_acc = 0, 0, 0
        for i, val_loader in enumerate(val_loaders):
            oIoU, mIoU, hit = validate(args, val_loader, model, local_rank)
            if i == 0: val_acc = mIoU 
            elif i == 1: testA_acc = mIoU 
            else: testB_acc = mIoU 
        if val_acc > best['val_acc'] and local_rank==0:
            if os.path.exists(best['path']):
                print('remove ', best['path'])
                os.remove(best['path'])
            save_path = save_checkpoint(epoch,model_without_ddp,optimizer,scheduler,logger,args, f'ckpt_320_epoch_{epoch}.pth')
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

        if local_rank == 0:
            writer.add_scalar('test/mIoU', val_acc, epoch)
            writer.add_scalar('test/hit', hit, epoch)
            writer.add_scalar('test/oIoU', oIoU, epoch)

    ### validate on the train dataset
    print()
    print()
    last_trainset = ReferDataset(refer_data_root=args.refer_data_root,
                            dataset=args.dataset,
                            split='train',
                            splitBy=args.splitBy,
                            image_transforms=get_transform(args.size, train=False),
                            eval_mode=True,
                            size=args.size,
                            bert_tokenizer=args.bert_tokenizer)
    val_train_loader = DataLoader(last_trainset,
                            batch_size=1,
                            num_workers=2,
                            pin_memory=True, 
                            sampler=val_sampler)
    print('loading ', best['path'])
    load_pretrained_checkpoint(best['path'], model_without_ddp)
    oIoU, mIoU, hit = validate(args, val_train_loader, model, local_rank)
    print('Validat on the train split: ', oIoU, mIoU, hit)
    print()
    print(best)
    #########

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(train_time))
    logger.info('Training + testing time {}'.format(total_time_str))

def sigmoid_mse_loss(input_logits, target_logits):
    """Takes sigmoid on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_sigmoid = F.sigmoid(input_logits)
    target_sigmoid = F.sigmoid(target_logits)

    return F.mse_loss(input_sigmoid, target_sigmoid, reduction='mean')

def train_one_epoch(train_loader,model,optimizer,epoch,local_rank,args, iteration=0, clip_model=None, model_ema=None):
    num_steps=len(train_loader)
    model.train()
    optimizer.zero_grad()

    batch_time=AverageMeter()
    loss_meter=AverageMeter()

    start=time.time()
    end=time.time()
    max_iter = int(num_steps * args.epoch) 

    if args.consistency_type == 'mse':
        # consistency_criterion = losses.softmax_mse_loss
        consistency_criterion = sigmoid_mse_loss
    elif args.consistency_type == 'kl':
        # consistency_criterion = losses.softmax_kl_loss
        consistency_criterion = F.kl_div

    for idx,(samples, targets) in enumerate(train_loader):
        word_ids = samples['word_ids'].squeeze(1)
        word_mask = samples['word_masks'].squeeze(1)
        img = samples['img'].cuda(local_rank,non_blocking=True)
        # target = targets['target'].cuda(local_rank,non_blocking=True)
        bbox = targets['boxes'].cuda(local_rank,non_blocking=True) 
        word_ids = word_ids.cuda(local_rank,non_blocking=True)
        word_mask = word_mask.cuda(local_rank,non_blocking=True)
        B = img.shape[0]
        pseudo = targets['pseudo_gt'].cuda(local_rank,non_blocking=True)

        output1, output2, output3, output4 = model(img, word_ids)
        if args.model_ema:
            with torch.no_grad():
                ema_output1, ema_output2, ema_output3, ema_output4 = model_ema(img, word_ids)
            ema_loss = 0
            ema_loss += consistency_criterion(output1, ema_output1)
            ema_loss += consistency_criterion(output2, ema_output2)
            ema_loss += consistency_criterion(output3, ema_output3)
            ema_loss += consistency_criterion(output4, ema_output4)
            l5 = ema_loss 
        else:
            l5 = torch.tensor(0) 

        l1 = criterion(output1, pseudo)
        l2 = criterion(output2, pseudo)
        l3 = criterion(output3, pseudo)
        l4 = criterion(output4, pseudo)

        loss = l1 + l2 + l3 + l4 + l5 

        # Synchronizes all processes.
        # all process statistic
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            try:
                model_ema.update(model)
            except:  # dog
                model_ema.update() 

        if args.distributed:
            reduce_tensor(args, l1)
            reduce_tensor(args, l2)
            reduce_tensor(args, l3)
            reduce_tensor(args, l4)
            reduce_tensor(args, l5)
            reduce_tensor(args, loss)

        if local_rank == 0:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('optim/lr', lr, iteration)
            writer.add_scalar('train/l1', l1.data.cpu().numpy(), iteration)
            writer.add_scalar('train/l2', l2.data.cpu().numpy(), iteration)
            writer.add_scalar('train/l3', l3.data.cpu().numpy(), iteration)
            writer.add_scalar('train/l4', l4.data.cpu().numpy(), iteration)
            writer.add_scalar('train/l5', l5.data.cpu().numpy(), iteration)
            writer.add_scalar('train/loss', loss.data.cpu().numpy(), iteration)

        # measure time
        loss_meter.update(loss.item(), img.size(0))
        batch_time.update(time.time()-end)
        end=time.time()

        if idx % args.print_freq==0 and local_rank==0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas=batch_time.avg*(num_steps-idx)
            all_etas = batch_time.avg * (max_iter - iteration)
            print(
                f'Train:[{epoch:2d}/{args.epoch}][{idx:4d}/{num_steps}] | '
                f'eta: {datetime.timedelta(seconds=int(etas))} | lr {lr:.6f} || '
                f'loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | '
                f'l1: {l1:.4f} | '
                f'l2: {l2:.4f} | '
                f'l3: {l3:.4f} | '
                f'l4: {l4:.4f} | '
                f'l5: {l5:.4f} | '
                f'time: {batch_time.val:.4f} ({batch_time.avg:.4f}) | '
                f'mem: {memory_used:.0f}MB || '
                f'all_eta: {datetime.timedelta(seconds=int(all_etas))}', flush=True)
        iteration += 1 
    epoch_time=time.time()-start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}", flush=True)
    return iteration 


if __name__=="__main__":
    torch.backends.cudnn.benchmark = True
    
    parse=get_parser()
    args=parse.parse_args()
    if args.vis_out is not None and not os.path.exists(args.vis_out):
        os.mkdir(args.vis_out)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank=int(os.environ['RANK'])
        world_size=int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank=-1
        world_size=-1

    print('='*100)
    print(args)
    print('='*100)

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()
    # 只在 rank 0 显示
    # cfg=config.get_config(args)
    if args.distributed:
        logger = create_logger(dist_rank=dist.get_rank())
    else:
        logger = create_logger()
    
    if args.board_folder is not None:
        global writer 
        writer = SummaryWriter(args.board_folder)

    main(args) 
