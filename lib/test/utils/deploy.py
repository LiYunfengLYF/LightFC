import os
from lib.test.evaluation.environment import env_settings


def get_onnx_save_name(params):
    save_dir = env_settings(params.env_num).save_dir
    onnx_save_dir = os.path.join(save_dir, "checkpoints/train/lightfc/%s" % (params.tracker_param))

    backbone_save_name = os.path.join(onnx_save_dir, 'deploy_lightTrack_ep%04d_backbone.onnx' % (params.cfg.TEST.EPOCH))
    network_save_name = os.path.join(onnx_save_dir, 'deploy_lightTrack_ep%04d_network.onnx' % (params.cfg.TEST.EPOCH))
    return {'backbone': backbone_save_name, 'network': network_save_name}


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()