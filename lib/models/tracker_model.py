import torch.nn as nn
from lib.models.lightfc import MobileNetV2, pwcorr_se_scf_sc_iab_sc_concat, repn33_se_center_concat
from lib.utils.load import load_pretrain


class LightFC(nn.Module):
    def __init__(self, cfg, env_num=0, training=False, ):
        super(LightFC, self).__init__()

        self.backbone = MobileNetV2()

        self.training = training
        if self.train:
            load_pretrain(self.backbone, env_num=env_num, training=training, cfg=cfg, mode=cfg.MODEL.BACKBONE.LOAD_MODE)

        self.fusion = pwcorr_se_scf_sc_iab_sc_concat(num_kernel=64,
                                                     adj_channel=96
                                                     )

        self.head = repn33_se_center_concat(inplanes=cfg.MODEL.HEAD.PARAMS.inplanes,
                                            channel=cfg.MODEL.HEAD.PARAMS.channel,
                                            feat_sz=cfg.MODEL.HEAD.PARAMS.feat_sz,
                                            stride=cfg.MODEL.HEAD.PARAMS.stride,
                                            freeze_bn=cfg.MODEL.HEAD.PARAMS.freeze_bn,
                                            )

    def forward(self, z, x, train=False):
        if self.training:
            z = self.backbone(z)
            x = self.backbone(x)

            opt = self.fusion(z, x)

            out = self.head(opt)
        else:
            return self.forward_tracking(z, x)
        return out

    #
    def forward_backbone(self, z):
        z = self.backbone(z)
        return z

    def forward_tracking(self, z_feat, x):
        x = self.backbone(x)
        opt = self.fusion(z_feat, x)
        out = self.head(opt)
        return out
