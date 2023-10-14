import torch
import torch.nn as nn

from lib.models.lightfc.head.erh import repN31


def pixel_wise_corr(z, x):
    '''
    z is kernel ([32, 96, 8, 8])
    x is search ([32, 96, 16, 16])

    z -> (32, 64, 96)
    x -> (32, 96, 256)
    '''
    b, c, h, w = x.size()
    z_mat = z.contiguous().view((b, c, -1)).transpose(1, 2)  # (b,64,c)
    x_mat = x.contiguous().view((b, c, -1))  # (b,c,256)
    return torch.matmul(z_mat, x_mat).view((b, -1, h, w))


class SE(nn.Module):

    def __init__(self, channels=64, reduction=1):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # nn.silu()
        return module_input * x


class pwcorr_se_scf_sc_iab_sc_concat(nn.Module):
    def __init__(self, num_kernel=64, adj_channel=96):
        super().__init__()

        # pw-corr
        self.pw_corr = pixel_wise_corr
        self.ca = SE()

        # SCF
        self.conv33 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=3, stride=1, padding=1,
                                groups=num_kernel)
        self.bn33 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=1, stride=1, padding=0,
                                groups=num_kernel)
        self.bn11 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # IAB
        self.conv_up = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel * 2, kernel_size=1, stride=1,
                                 padding=0)
        self.bn_up = nn.BatchNorm2d(num_kernel * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()

        self.conv_down = nn.Conv2d(in_channels=num_kernel * 2, out_channels=num_kernel, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_down = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.adjust = nn.Conv2d(num_kernel, adj_channel, 1)

    def forward(self, z, x):
        corr = self.ca(self.pw_corr(z, x))

        # scf + skip-connection
        corr = corr + self.bn11(self.conv11(corr)) + self.bn33(self.conv33(corr))

        # iab + skip-connection
        corr = corr + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr)))))

        corr = self.adjust(corr)

        corr = torch.cat((corr, x), dim=1)

        return corr


class pwcorr_se_repn31_sc_iab_sc_adj_concat(nn.Module):
    def __init__(self, num_kernel=64, adj_channel=96):
        super().__init__()
        self.pw_corr = pixel_wise_corr
        self.ca = SE()

        # SCF reparam structure
        self.repn31 = repN31(num_kernel, num_kernel, kernel_size=3, padding=1, groups=num_kernel,nonlinearity=nn.Identity)

        # IAB
        self.conv_up = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel * 2, kernel_size=1, stride=1,
                                 padding=0)
        self.bn_up = nn.BatchNorm2d(num_kernel * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()
        self.conv_down = nn.Conv2d(in_channels=num_kernel * 2, out_channels=num_kernel, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_down = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # adj layer
        self.adjust = nn.Conv2d(num_kernel, adj_channel, 1)

    def forward(self, z, x):
        corr = self.ca(self.pw_corr(z, x))

        corr = corr + self.repn31(corr)

        corr = corr + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr)))))

        corr = self.adjust(corr)

        corr = torch.cat((corr, x), dim=1)
        return corr
