import torch

from lib.models import LightFC
from lib.utils.box_ops import clip_box, box_xywh_to_xyxy, box_iou, box_xyxy_to_xywh
from lib.test.utils.hann import hann2d
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target


class lightFC(BaseTracker):
    def __init__(self, params, dataset_name):
        super(lightFC, self).__init__(params)

        network = LightFC(cfg=params.cfg, env_num=None, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)

        for module in network.backbone.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        for module in network.head.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.frame_id = 0

    def initialize(self, image, info: dict):
        H, W, _ = image.shape

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            self.z_feat = self.network.forward_backbone(template.tensors)

        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward_tracking(z_feat=self.z_feat, x=x_dict.tensors)

        response_origin = self.output_window * out_dict['score_map']

        pred_box_origin = self.compute_box(response_origin, out_dict,
                                           resize_factor).tolist()  # .unsqueeze(dim=0)  # tolist()

        self.state = clip_box(self.map_box_back(pred_box_origin, resize_factor), H, W, margin=2)

        return {"target_bbox": self.state}

    def compute_box(self, response, out_dict, resize_factor):
        pred_boxes = self.network.head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_boxes = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor)
        return pred_boxes

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return lightFC
