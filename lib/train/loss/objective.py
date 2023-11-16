import torch
from .focal_loss import FocalLoss
from .box_loss import giou_loss, ciou_loss, siou_loss, eiou_loss, wiou_loss
from torch.nn.functional import l1_loss, smooth_l1_loss
import torch.nn as nn

from .gfocal_loss import DistributionFocalLoss
from .varifocal_loss import VarifocalLoss


class lightTrackObjective(object):
    def __init__(self, cfg):
        super(lightTrackObjective, self).__init__()

        # l loss
        if cfg.TRAIN.L_LOSS == 'l1':
            self.l1 = l1_loss
        elif cfg.TRAIN.L_LOSS == 'smooth_l1':
            self.smooth_l1 = smooth_l1_loss
        else:
            pass

        # box iou
        if cfg.TRAIN.BOX_LOSS == 'giou':
            self.iou = giou_loss
        elif cfg.TRAIN.BOX_LOSS == 'ciou':
            self.iou = ciou_loss
        elif cfg.TRAIN.BOX_LOSS == 'siou':
            self.iou = siou_loss
        elif cfg.TRAIN.BOX_LOSS == 'wiou':
            self.iou = wiou_loss
        elif cfg.TRAIN.BOX_LOSS == 'eiou':
            self.iou = eiou_loss
        else:
            pass

        # cls iou
        if cfg.TRAIN.CLS_LOSS == 'focal':
            self.focal_loss = FocalLoss()
        elif cfg.TRAIN.CLS_LOSS == 'varifocal':
            self.focal_loss = VarifocalLoss()

