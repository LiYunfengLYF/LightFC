import torch
import time

from torch import nn

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


if __name__ == "__main__":
    # test the running speed

    use_gpu = True
    z_feat = torch.rand(1, 96, 8, 8).cuda()
    x = torch.rand(1, 3, 256, 256).cuda()

    if use_gpu:
        model = lightTrack_track().cuda()
        x = x.cuda()
        z_feat = z_feat.cuda()
    # oup = model(x, zf)

    T_w = 10  # warmup
    T_t = 100  # test
    with torch.no_grad():
        for i in range(T_w):
            oup = model(z_feat, x)
        t_s = time.time()
        for i in range(T_t):
            oup = model(z_feat, x)
        t_e = time.time()
        print('speed: %.2f FPS' % (T_t / (t_e - t_s)))
