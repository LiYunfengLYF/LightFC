import os
import yaml
import torch
from easydict import EasyDict as edict
from lib.train.admin.local import EnvironmentSettings as env


def load_pretrain(backbone, env_num=0, training=True, mode=1, cfg=None):
    pretrained_path = env(env_num=env_num).pretrained_networks
    if cfg.MODEL.BACKBONE.PRETRAIN_FILE is not None and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.BACKBONE.PRETRAIN_FILE)
    else:
        pretrained = ''

    if training and cfg.MODEL.BACKBONE.USE_PRETRAINED:
        print(f'Try Loading Pretrained Model, using mode {mode}')
        try:

            if mode == 1:
                checkpoint = torch.load(pretrained, map_location="cpu")
            elif mode == 2:
                checkpoint = torch.load(pretrained, map_location="cpu")['state_dict']
            elif mode == 3:
                checkpoint = torch.load(pretrained, map_location="cpu")['model']
            else:
                raise
            missing_keys, unexpected_keys = backbone.load_state_dict(checkpoint, strict=True)
            print('Load pretrained model from: ' + pretrained)
        except Exception as e:
            print(e)
        print('Loading Finish ....')


def load_yaml(yaml_file):
    with open(yaml_file) as f:
        exp_config = yaml.safe_load(f)
    exp_config = edict(exp_config)
    return exp_config
