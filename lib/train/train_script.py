import importlib
import os
import random

import numpy as np

from lib.models import *
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.train.loss.box_loss import giou_loss

from torch.nn.functional import l1_loss
from lib.models.ostrack import build_ostrack
from lib.train.actors import *
from lib.train.actors.ostrack import OSTrackActor
from lib.train.data.base_functions import *
from lib.train.loss.focal_loss import FocalLoss
from lib.train.trainers import LTRTrainer
from lib.models.lightfc.trackerModel import TRACKER_REGISTRY
from lib.models.lightTrackST.trackerModel import TRACKER_REGISTRY
from lib.utils.load import load_yaml
from lib.train.loss import lightTrackObjective


def run(settings):
    settings.description = 'Training script'

    cfg = load_yaml(settings.cfg_file)
    print('CFG', cfg)
    update_settings(settings, cfg)

    # init seed
    random.seed(cfg.TRAIN.SEED)
    np.random.seed(cfg.TRAIN.SEED)
    torch.manual_seed(cfg.TRAIN.SEED)
    torch.cuda.manual_seed(cfg.TRAIN.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)
    _, loader_val = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name == "lightfc":
        net = TRACKER_REGISTRY.get('lightfc')(cfg, env_num=settings.env_num, training=True)
    elif settings.script_name == "lightfc_st":
        net = TRACKER_REGISTRY.get('lightfc_st')(cfg, env_num=settings.env_num, training=True)
    elif settings.script_name == "ostrack":
        net = build_ostrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    # Actors
    if settings.script_name == "lightfc":
        objective = lightTrackObjective(cfg)
        loss_weight = {'iou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': cfg.TRAIN.LOC_WEIGHT, }
        actor = lightTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    elif settings.script_name == "lightfc_st":
        objective = lightTrackObjective(cfg)
        loss_weight = {'iou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': cfg.TRAIN.LOC_WEIGHT, }
        actor = lightTrackSTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    elif settings.script_name == "ostrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.}
        actor = OSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    else:
        raise ValueError("illegal script name")

    # SWA
    settings.use_swa = getattr(cfg.TRAIN, 'USE_SWA', False)
    settings.swa_epoch = getattr(cfg.TRAIN, 'SWA_EPOCH', None)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp, )

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
