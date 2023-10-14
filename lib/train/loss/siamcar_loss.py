import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_locations(features, stride):
    h, w = features.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride,
        features.device
    )
    return locations_per_level


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride + 3*stride  # (size_z-1)/2*size_z 28
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride
    locations = torch.stack((shift_x, shift_y), dim=1) + 32  # alex:48 // 32
    return locations


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class SiamCARLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self):
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets = self.compute_targets_for_locations(
            points, labels, gt_bbox
        )

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, labels, gt_bbox):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        labels = labels.view(16 ** 2, -1)

        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        s1 = reg_targets_per_im[:, :, 0] > 0.6 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.6 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        is_in_boxes = s1 * s2 * s3 * s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1

        return labels.permute(1, 0).contiguous(), reg_targets_per_im.permute(1, 0, 2).contiguous()

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def compute_locations(self, cls, stride):
        return compute_locations(cls, stride)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        print(pos_inds)
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


if __name__ == '__main__':
    loss = SiamCARLossComputation()
