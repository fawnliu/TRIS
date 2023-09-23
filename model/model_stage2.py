import os
# os.environ['CUDA_ENABLE_DEVICES'] = '6'

import torch 
import torch.nn as nn
import torch.nn.functional as F
import CLIP.clip as clip 

from model.attn import PixelAttention

class ConvBNRelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, G=32, use_relu=True):
        super(ConvBNRelu, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.use_relu:
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)



class TRIS(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        self.clip = ('clip' in args.bert_tokenizer)

        # print('bert_tokenizer = ', args.bert_tokenizer)
        # print('clip = ', self.clip)
        # print(args.backbone)
        
        type = args.backbone.split('-')[-1]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load(type, device=device, jit=False, txt_length=args.max_query_len)
        # clip_model = clip_model.eval().float() 
        clip_model = clip_model.float() 
        self.backbone = clip_model 

        self.textdim = 512 

        ################################################################################################################
        if type == 'RN50':
            v_chans = [256, 512, 1024, 2048]  # 1024 or 2048, attention pool 
        elif type == 'RN101':
            v_chans = [256, 512, 1024, 2048]
        l_chans = self.textdim 

        self.attention2 = PixelAttention(visual_channel=v_chans[1], language_channel=l_chans)
        self.attention3 = PixelAttention(visual_channel=v_chans[2], language_channel=l_chans)
        self.attention4 = PixelAttention(visual_channel=v_chans[3], language_channel=l_chans)
        ################################################################################################################
        self.reduced_c1 = ConvBNRelu(v_chans[0], 64, kernel_size=3, stride=1, padding=1)
        self.reduced_c2 = ConvBNRelu(v_chans[1], 128, kernel_size=3, stride=1, padding=1)
        self.reduced_c3 = ConvBNRelu(v_chans[2], 256, kernel_size=3, stride=1, padding=1)
        self.reduced_c4 = ConvBNRelu(v_chans[3], 512, kernel_size=3, stride=1, padding=1)
        ##################
        self.final_seg1 = nn.Sequential(
            ConvBNRelu(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.final_seg2 = nn.Sequential(
            ConvBNRelu(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.final_seg3 = nn.Sequential(
            ConvBNRelu(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self.final_seg4 = nn.Sequential(
            ConvBNRelu(256, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))
        # ##################
        self.output4 = ConvBNRelu(512, 256, kernel_size=3, stride=1, padding=1)
        self.output3 = ConvBNRelu(256, 128, kernel_size=3, stride=1, padding=1)
        self.output2 = ConvBNRelu(128, 64, kernel_size=3, stride=1, padding=1)
        self.output1 = ConvBNRelu(64, 32, kernel_size=3, stride=1, padding=1)     

    
    def trainable_parameters(self):
        # print(self.parameters())
        backbone = []
        head = []
        for k, v in self.named_parameters():
            if k.startswith('backbone') and 'positional_embedding' not in k:
                backbone.append(v)
            else:
                head.append(v)
        print('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
        return backbone, head 

    def forward(self, x, word_id): 
        img_size = x.shape[2:]
        _,_,H,_ = x.size()
        
        word_embedding, hidden = self.backbone.encode_text(word_id) 
        # c1, c2, c3, c4 = self.backbone.encode_image(x)  
        c1, c2, c3, c4, attn_out = self.backbone.encode_image(x)  

        lan = word_embedding.permute(0, 2, 1)  # [N, T, C] -> [N, C, T]

        fuse_feats2 = self.attention2(c2, lan) + c2 
        fuse_feats3 = self.attention3(c3, lan) + c3 
        fuse_feats4 = self.attention4(c4, lan) + c4 

        dem1 = self.reduced_c1(c1)
        dem2 = self.reduced_c2(fuse_feats2)
        dem3 = self.reduced_c3(fuse_feats3)
        dem4 = self.reduced_c4(fuse_feats4)

        seg_out4 = Upsample(self.output4(dem4), dem3.shape[2:])  # 512 -> 256  
        seg_out3 = Upsample(self.output3(seg_out4 + dem3), dem2.shape[2:])  # [2, 128, 40, 40]
        seg_out2 = Upsample(self.output2(seg_out3 + dem2), dem1.shape[2:])  # [2, 64, 80, 80]
        seg_out1 = self.output1(seg_out2 + dem1)
        seg_final_out1 = Upsample(self.final_seg1(seg_out1), img_size)
        
        if self.training:
            seg_final_out4 = Upsample(self.final_seg4(seg_out4), img_size)
            seg_final_out3 = Upsample(self.final_seg3(seg_out3), img_size)
            seg_final_out2 = Upsample(self.final_seg2(seg_out2), img_size)
            return seg_final_out1, seg_final_out2, seg_final_out3, seg_final_out4 
            # return seg_final_out1, seg_final_out1, seg_final_out1, seg_final_out1
        else:
            return seg_final_out1 

def criterion(seg_final_out1, target):
    target = target.float() 
    return F.binary_cross_entropy_with_logits(input=seg_final_out1, target=target, reduction='mean')

