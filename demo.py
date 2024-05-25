import os
import torch
# from dataset.transform import get_transform
from args import get_parser
from model.model_stage2 import TRIS 
import cv2 
import numpy as np 
import CLIP.clip as clip 
from PIL import Image 
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def get_transform(size=None):
    if size is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def visualize_cam(normalized_heatmap, original=None, root=None):
    map_img = np.uint8(normalized_heatmap * 255)
    heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
    if original is not None:
        original_img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        img = cv2.addWeighted(heatmap_img, .6, original_img, 0.4, 0)
    else:
        img = heatmap_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if root is None:
        return img 
    plt.imsave(root, img)

def get_norm_cam(cam):
    cam = torch.clamp(cam, min=0)
    cam_t = cam.unsqueeze(0).unsqueeze(0).flatten(2)
    cam_max = torch.max(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
    cam_min = torch.min(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
    norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-5)
    norm_cam = norm_cam.squeeze(0).squeeze(0).cpu().numpy()
    return norm_cam

def prepare_data(img_path, text, max_length=20):
    img = cv2.imread(img_path)

    word_ids = []
    split_text = text.split(',')
    tokenizer = clip.tokenize
    for text in split_text:
        word_id = tokenizer(text).squeeze(0)[:max_length]
        word_ids.append(word_id.unsqueeze(0))
    word_ids = torch.cat(word_ids, dim=-1)

    h, w, c = img.shape

    img = Image.fromarray(img)
    transform = get_transform(size=img_size)
    img = transform(img)
    # word_ids = torch.tensor(word_ids)

    return img, word_ids, h, w

if __name__ == '__main__':
    import os 
    os.environ['CUDA_ENABLE_DEVICES'] = '0'
    parse=get_parser()
    args=parse.parse_args()
    img_size = 320 
    max_length = 20 

    model=TRIS(args)
    model.cuda()

    model_path = 'weights/stage2_refcocog_google.pth' 
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)

    img_path = args.img 
    text = args.text 
    img, word_id, h, w = prepare_data(img_path, text, max_length)
    word_id = word_id.view(-1)

    outputs = model(img.unsqueeze(0).cuda(), 
                    word_id.unsqueeze(0).cuda())
    
    output = outputs[0] 
    pred = F.interpolate(output, (h,w), align_corners=True, mode='bilinear').squeeze(0).squeeze(0)
    
    norm_cam = get_norm_cam(pred.detach().cpu())
    orig_img = cv2.imread(args.img)

    ## save to f"figs/demo_({text}).png"
    visualize_cam(norm_cam, orig_img, root=f"figs/demo_({text}).png")

    # python demo.py  --img figs/demo.png  --text 'man on the right'

