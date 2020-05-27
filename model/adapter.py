'''
Draft Version
'''

import torch.nn as nn
import torch
from torch.nn.init import kaiming_uniform_


class Adapter(nn.Module):
    def __init__(self, Cfgs):
        super(Adapter, self).__init__()
        self.Cfgs = Cfgs
        #imgfeat_linear_size = Cfgs.FEAT_SIZE['clevr']['GRID_FEAT'][1]
        self.conv = nn.Sequential(nn.Conv2d(1024, Cfgs.HIDDEN_SIZE, 3, padding = 1),
                                  nn.ELU(),
                                  nn.Conv2d(Cfgs.HIDDEN_SIZE, Cfgs.HIDDEN_SIZE, 3, padding = 1),
                                  nn.ELU())
        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

    def forward(self, grid_feat):
        #grid_feat = feat_dict['GRID_FEAT']
        img_feat = grid_feat.permute(0, 2, 1)
        img_feat = img_feat.view(-1, 1024, 14, 14)
        img_feat = self.conv(img_feat)

        return img_feat