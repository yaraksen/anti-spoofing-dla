from torch import nn
import torch
from typing import List
from src.model.rawnet2.sinc_conv import SincFilters
from src.model.rawnet2.res_block import ResBlock


class RawNet2(nn.Module):
    def __init__(
        self,
        small_resblock_kernel_size: int,
        large_resblock_kernel_size: int,
        norm_before_gru: bool,
        gru_hidden: int,
        gru_num_layers: int,
        **kwargs
    ):
        super(self).__init__()
        assert kwargs["sinc_out_channels"] == small_resblock_kernel_size

        self.sinc_filter = SincFilters(**kwargs)
        self.resblocks = nn.Sequential(
            # small resblocks
            ResBlock(small_resblock_kernel_size, small_resblock_kernel_size),
            ResBlock(small_resblock_kernel_size, small_resblock_kernel_size),
            # large resblocks
            ResBlock(small_resblock_kernel_size, large_resblock_kernel_size),
            ResBlock(large_resblock_kernel_size, large_resblock_kernel_size),
            ResBlock(large_resblock_kernel_size, large_resblock_kernel_size),
            ResBlock(large_resblock_kernel_size, large_resblock_kernel_size),
        )

        if norm_before_gru:
            self.bn_act_gru = nn.Sequential(
                nn.BatchNorm1d(large_resblock_kernel_size), nn.LeakyReLU(0.3)
            )
        else:
            self.bn_act_gru = nn.Identity()

        self.gru = nn.GRU(
            input_size=large_resblock_kernel_size,
            hidden_size=gru_hidden,
            num_layers=gru_num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(gru_hidden, gru_hidden)
        self.fc2 = nn.Linear(gru_hidden, 2)

    def forward(self, audio, **batch):
        x = self.sinc_filter(audio)
        x = self.resblocks(x)
        x = self.gru(self.bn_act_gru(x).transpose(1, 2))[0][:, -1]
        x = self.fc1(x)
        x = x / (0.1 * torch.norm(x, p=2, dim=1, keepdim=True))
        x = self.fc2(x)
        return x
