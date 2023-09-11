import numpy as np 
import torch 
import torch.nn as nn
# import transformers
from transformers import RobertaModel, BertModel, GPT2LMHeadModel
import torch.nn.functional as F

import pdb 
import CLIP.clip as clip 

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)

            