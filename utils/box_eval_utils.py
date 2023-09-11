import torch
import numpy as np

from utils.box_utils import box_iou
import cv2 
import torchvision 
import matplotlib.pyplot as plt


def eval_box_iou(pred_boxes, gt_boxes):
    iou, union = box_iou(pred_boxes, gt_boxes)
    iou = torch.diag(iou)
    iou = torch.sum(iou) 
    return iou

def eval_box_acc(pred_boxes, gt_boxes):
    for bbox_p in pred_boxes:
        bbox_p = torch.tensor(bbox_p[0:4]).unsqueeze(0)
        iou, union = box_iou(bbox_p, gt_boxes)
        iou = torch.diag(iou)
        iou = torch.sum(iou) 
        if iou > 0.5:
            return torch.tensor(1)
    return torch.tensor(0)

def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def generate_bbox(cam, threshold=0.5, nms_threshold=0.05, max_drop_th=0.5):
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

    thr_val = threshold * np.max(gray_heatmap)

    _, thr_gray_heatmap = cv2.threshold(gray_heatmap,
                                        int(thr_val), 255,
                                        cv2.THRESH_TOZERO)
    try:
        _, contours, _ = cv2.findContours(thr_gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        proposals = [cv2.boundingRect(c) for c in contours]
        # proposals = [(x, y, w, h) for (x, y, w, h) in proposals if h * w > 0.05 * 224 * 224]
        if len(proposals) > 0:
            proposals_with_conf = [thr_gray_heatmap[y:y + h, x:x + w].mean()/255 for (x, y, w, h) in proposals]
            inx = torchvision.ops.nms(torch.tensor(proposals).float(),
                                      torch.tensor(proposals_with_conf).float(),
                                      nms_threshold)
            estimated_bbox = torch.cat((torch.tensor(proposals).float()[inx],
                                        torch.tensor(proposals_with_conf)[inx].unsqueeze(dim=1)),
                                       dim=1).tolist()
            estimated_bbox = [(x, y, x+w, y+h, conf) for (x, y, w, h, conf) in estimated_bbox
                              if conf > max_drop_th * np.max(proposals_with_conf)]
        else:
            estimated_bbox = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]]
    else:
        estimated_bbox = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]]
    return estimated_bbox


