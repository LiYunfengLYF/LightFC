import torch
import torch.nn as nn
from . import BaseActor
from ..loss.cos_sim_loss import cosine_similarity_loss
from ...utils.box_ops import box_xywh_to_xyxy, box_cxcywh_to_xyxy
from ...utils.heapmap_utils import generate_heatmap
import torchvision.transforms as transforms
'''
vot initialize vot2020 --workspace /home/liyunfeng/coda2/tracker-lightweight/external/vot20
'''

class lightTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        # triple loss
        # self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.triple = nn.TripletMarginLoss(margin=1, p=2, reduction='mean')

        # self.transform = transforms.RandomErasing(p=0.05, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0, inplace=False)

    def __call__(self, data):

        out_dict = self.forward_pass(data)

        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][:, i, :].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        search_img = data['search_images'][:, 0, :].view(-1, *data['search_images'].shape[2:])
        # search_img = self.transform(search_img)

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(z=template_list, x=search_img)
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        bs, n, _ = gt_dict['search_anno'].shape
        gt_bbox = gt_dict['search_anno'].view(bs, 4)

        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'].view(n, bs, 4), self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps_flatten = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        # locate box
        try:
            iou_loss, iou = self.objective.iou(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            iou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # l1 loss
        l1_loss = self.objective.l1(pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        if 'score_map' in pred_dict:
            location_loss = self.objective.focal_loss(pred_dict['score_map'], gt_gaussian_maps_flatten)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # cos_sim_loss
        # try:
        #     compute_tri_loss = self.compute_tri_loss(bs, pred_dict, gt_gaussian_maps_flatten)
        # except:
        #     compute_tri_loss = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        loss = self.loss_weight['iou'] * iou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'focal'] * location_loss  # + compute_tri_loss * 0.05
        # * location_loss  # cos_sim_loss * 0.1

        # return
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": iou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      # "Loss/cossim": cos_sim_loss.item(),
                      # "Loss/triple": compute_tri_loss.item(),
                      "mean_IoU": mean_iou.item(),
                      }
            return loss, status
        else:
            return loss

    # def compute_cos_sim_loss(self, bs, pred_dict, gt_gaussian_maps):
    #
    #     cos_sim_loss_list = []
    #     for i in range(bs):
    #         # Negtive
    #         response_item = (pred_dict['score_map'][i] + CORE1_REGION.to(pred_dict['score_map'].device))
    #         response_item_flatten = response_item.flatten(1)
    #         topk_values, topk_indices = torch.topk(response_item_flatten, k=5, largest=True)
    #
    #         max_value = torch.max(pred_dict['score_map'][i])
    #
    #         # print(pred_dict['opt_feat']['cls'].shape)
    #         feature_item = pred_dict['opt_feat']['cls'][i]
    #         feature_item_flatten = feature_item.flatten(1)
    #
    #         # Postive
    #         gt_item = gt_gaussian_maps[i]
    #         gt_item_flatten = gt_item.flatten(1)
    #         _, gt_idx = torch.max(gt_item_flatten, dim=1, keepdim=True)
    #         pos = feature_item_flatten[:, gt_idx].flatten(0).unsqueeze(dim=0)
    #
    #         for id, indice in enumerate(topk_indices[0]):
    #             # print(topk_values[0][id])
    #             if float(topk_values[0][id]) < 0.7 * max_value:
    #                 continue
    #             ne = feature_item_flatten[:, indice].unsqueeze(dim=0)
    #             loss = cosine_similarity_loss(pos, ne, margin=0.0)
    #             cos_sim_loss_list.append(loss)
    #     return sum(cos_sim_loss_list) / len(cos_sim_loss_list)
    #
    # def compute_tri_loss(self, bs, pred_dict, gt_gaussian_maps):
    #
    #     tri_loss_list = []
    #     for i in range(bs):
    #         # idx
    #         response_item = (pred_dict['score_map'][i] + CORE1_REGION.to(pred_dict['score_map'].device))
    #         response_item_flatten = response_item.flatten(1)
    #         topk_values, topk_indices = torch.topk(response_item_flatten, k=5, largest=True)
    #
    #         max_value = torch.max(pred_dict['score_map'][i])
    #
    #         feature_item = pred_dict['x_feat'][i]
    #         feature_item_flatten = feature_item.flatten(1)
    #
    #         # Anchor
    #         anchor = self.avg_pooling(pred_dict['z_feat']).flatten(2).squeeze(dim=2)[i]
    #
    #         # Postive
    #         gt_item = gt_gaussian_maps[i]
    #         gt_item_flatten = gt_item.flatten(1)
    #         _, gt_idx = torch.max(gt_item_flatten, dim=1, keepdim=True)
    #         pos = feature_item_flatten[:, gt_idx].flatten(0)  # .unsqueeze(dim=0)
    #
    #         for id, indice in enumerate(topk_indices[0]):
    #             if float(topk_values[0][id]) < 0.7 * max_value:
    #                 continue
    #             ne = feature_item_flatten[:, indice]
    #             tri_loss_list.append(self.triple(anchor, pos, ne))
    #     return sum(tri_loss_list) / len(tri_loss_list)

# CORE1_REGION = torch.tensor([
#     [
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
#     ]
# ])
