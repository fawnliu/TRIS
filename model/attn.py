from turtle import forward
import torch 
import torch.nn as nn 
from torch import Tensor, cartesian_prod, normal
from torch.nn import functional as F 
import math


class PixelAttention(nn.Module):
    #  https://github.com/yz93/LAVT-RIS
    def __init__(
        self,
        visual_channel, # input visual features' channel
        language_channel, # input language features,
    )->None:
    
        super().__init__()
        self.Ci = visual_channel
        self.Ct = language_channel

        # convolution op
        # Ct  = > Ci 
        self.Wk = nn.Conv1d(self.Ct, self.Ci, 1)
        self.Wv = nn.Conv1d(self.Ct, self.Ci, 1)
        # Ci  = > Ci
        self.Wq = nn.Conv2d(self.Ci, self.Ci, 1)
        self.Wm = nn.Conv2d(self.Ci, self.Ci, 1)
        self.Ww = nn.Conv2d(self.Ci, self.Ci, 1)
        self.Wo = nn.Conv2d(self.Ci, self.Ci, 1)
        
        # instance normalization
        self.ins_q = nn.InstanceNorm2d(self.Ci, affine=True)
        self.ins_w = nn.InstanceNorm2d(self.Ci, affine=True)
        # self.ins_q = nn.BatchNorm2d(self.Ci)
        # self.ins_w = nn.BatchNorm2d(self.Ci)

    def forward(self, vis_feat:Tensor, lan_feat:Tensor):
        """
        Input:
            vis_feat: 
                Visual Features from each stage [N,Ci,H,W]
            lan_feat:
                Language features from BERT Encoder [N,Ct,T]
        Output:
            output_features: [N,Ci,H,W]
        """
        N, Ci, H, W = vis_feat.size()
        N, Ct, T = lan_feat.size()
        Lk, Lv = self.Wk(lan_feat), self.Wv(lan_feat) # [N,Ci,T]
        Vq = self.ins_q(self.Wq(vis_feat)) # [N,Ci,H,W]
        
        Vq = Vq.view(N,Ci,H*W).permute(0,2,1) # [N,H*W,Ci]
        # get attention map 
        attn = F.softmax(Vq.matmul(Lk) / math.sqrt(Ci), dim=2) # [N,H*W,T]

        Lv = Lv.permute(0, 2, 1) #[N,T,Ci]
        G = attn.matmul(Lv) # [N,H*W,Ci]

        G = G.permute(0, 2, 1).view(N, Ci, H, W) # [N,Ci,H,W]
        Gi = self.ins_w(self.Ww(G)) # [N,Ci,H,W]

        Vo = F.relu(self.Wm(vis_feat)) # [N,Ci,H,W]
        out_feat = F.relu(self.Wo(Vo * Gi)) # [N,Ci,H,W]

        return out_feat


class bilateral_prompt(nn.Module):
    def __init__(self, vis_chans, lan_chans, m_chans=None) -> None:
        super().__init__()
        if m_chans is None:
            m_chans = vis_chans 
        self.v_proj1 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True)
        )
        self.v_proj2 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True)
        )
        self.v_proj3 = nn.Sequential(
            nn.Conv2d(vis_chans, m_chans, 1),
            nn.InstanceNorm2d(m_chans, affine=True),
            nn.ReLU(inplace=True)
        )

        self.t_proj1 = nn.Sequential(
            nn.Linear(lan_chans, m_chans),
            nn.ReLU(inplace=True)
        )
        self.t_proj2 = nn.Sequential(
            nn.Linear(lan_chans, m_chans),
            nn.ReLU(inplace=True)
        )
        self.t_proj3 = nn.Sequential(
            nn.Linear(lan_chans, m_chans),
            nn.ReLU(inplace=True)
        )

        self.v_output = nn.Sequential(
            nn.Conv2d(m_chans, vis_chans, 1),
            nn.InstanceNorm2d(vis_chans, affine=True)
        )
        
        self.t_output = nn.Sequential( 
            nn.Linear(m_chans, lan_chans)
        )
    
    def forward(self, vis, lan):
        B, C, H, W = vis.shape
        lan = lan.transpose(1, 2)
        B, N ,C = lan.shape 

        Ci = lan.shape[-1]

        Qv, Kv, Vv = self.v_proj1(vis), self.v_proj2(vis), self.v_proj3(vis)
        Qt, Kt, Vt = self.t_proj1(lan), self.t_proj2(lan), self.t_proj3(lan)

        Qv = Qv.reshape(B, C, -1).transpose(1,2)
        Av = F.softmax(Qv.matmul(Kt.transpose(1, 2)) / math.sqrt(Ci), dim=2)  

        Kv = Kv.reshape(B, C, -1)
        At = F.softmax(Qt.matmul(Kv) / math.sqrt(Ci), dim=2)  

        new_vis = Av.matmul(Vt)  
        
        Vv = Vv.reshape(B, C, -1).transpose(1, 2)
        new_lan = At.matmul(Vv)  

        new_vis = new_vis.permute(0, 2, 1).reshape(B, C, H, W)

        new_vis = self.v_output(new_vis)
        new_lan = self.t_output(new_lan)
        return new_vis, new_lan 



if __name__=="__main__":
    vis_feat=torch.rand(4,32,64,64)
    lan_feat=torch.rand(4,128,20)
    pwan=PixelWordAttention(32,128)
    print(pwan(vis_feat,lan_feat).size())








