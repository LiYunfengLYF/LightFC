import argparse

import torch
import torch.nn as nn
from thop import profile, clever_format

from lib.models.lightfc import MobileNetV2, repn33_se_center_concat
from lib.models.lightfc.fusion.ecm import pwcorr_se_repn31_sc_iab_sc_adj_concat


class lightTrack_track(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = MobileNetV2()
        self.fusion = pwcorr_se_repn31_sc_iab_sc_adj_concat()
        self.head = repn33_se_center_concat(inplanes=192, channel=256)

        for module in self.backbone.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        for module in self.fusion.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        for module in self.head.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

    def forward(self, z, x):
        x = self.backbone(x)
        opt = self.fusion(z, x)
        out = self.head(opt)

        return out



model = lightTrack_track().cuda().eval()

if __name__ == '__main__':
    z_feat = torch.rand(1, 96, 8, 8).cuda()
    x = torch.rand(1, 3, 256, 256).cuda()
    macs, params = profile(model, inputs=(z_feat, x), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
