import torch
from torchvision.ops.boxes import box_area
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
'''Note that this function only supports shape (N,4)'''


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]


def batch_xywh2center(boxes):
    cx = boxes[:, 0] + (boxes[:, 2] - 1) / 2
    cy = boxes[:, 1] + (boxes[:, 3] - 1) / 2
    w = boxes[:, 2]
    h = boxes[:, 3]

    if isinstance(boxes, np.ndarray):
        return np.stack([cx, cy, w, h], 1)
    else:
        return torch.stack([cx, cy, w, h], 1)


def batch_xywh2center2(boxes):
    cx = boxes[:, 0] + boxes[:, 2] / 2
    cy = boxes[:, 1] + boxes[:, 3] / 2
    w = boxes[:, 2]
    h = boxes[:, 3]

    if isinstance(boxes, np.ndarray):
        return np.stack([cx, cy, w, h], 1)
    else:
        return torch.stack([cx, cy, w, h], 1)

def batch_xywh2corner(boxes):
    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 0] + boxes[:, 2]
    ymax = boxes[:, 1] + boxes[:, 3]

    if isinstance(boxes, np.ndarray):
        return np.stack([xmin, ymin, xmax, ymax], 1)
    else:
        return torch.stack([xmin, ymin, xmax, ymax], 1)