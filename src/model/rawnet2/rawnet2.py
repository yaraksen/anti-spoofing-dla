from torch import nn
import torch
from typing import List
from src.model.rawnet2.sinc_conv import SincFilters
from src.model.rawnet2.res_block import ResBlock


class RawNet2(nn.Module):
    def __init__(self, **kwargs):
        super(self).__init__()
        self.sinc_filter = SincFilters(**kwargs)
        self.big_resblocks = self.get_resblocks(**kwargs)
        self.small_resblocks = self.get_resblocks(**kwargs)
    
    def get_resblocks(self, **kwargs):
        pass

    def forward(self, audio, **batch):
        pass
        # return {"logits": out}
