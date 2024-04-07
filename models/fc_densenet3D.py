import lightning as l
from models.layers import *
from utils.utils import get_argparser_group
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os

class FCDenseNetEncoder(l.LightningModule):
    def __init__(self, in_channels, down_blocks, bottleneck_layers, growth_rate, out_chans_first_conv):
        super().__init__()
        self.down_blocks = down_blocks
        self.skip_connection_channel_counts = []
        self.has_bottle_neck = True if bottleneck_layers>0 else False
        self.firstconv = nn.Conv2d(in_channels, out_chans_first_conv, kernel_size=3, stride=1, padding=1, bias=True)
        self.cur_channels_count = out_chans_first_conv

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(DenseBlock(self.cur_channels_count, growth_rate, down_blocks[i]))
            self.cur_channels_count += (growth_rate * down_blocks[i])
            self.skip_connection_channel_counts.insert(0, self.cur_channels_count)
            self.transDownBlocks.append(TransitionDown(self.cur_channels_count))
        if self.has_bottle_neck:
            self.bottleneck = Bottleneck(self.cur_channels_count, growth_rate, bottleneck_layers)
        self.prev_block_channels = growth_rate * bottleneck_layers
        self.cur_channels_count += self.prev_block_channels

    def forward(self, x):
        out = self.firstconv(x)
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
        if self.has_bottle_neck:
            out = self.bottleneck(out)
        return out, skip_connections


class FCDenseNetDecoder(l.LightningModule):
    def __init__(self, prev_block_channels, skip_connection_channel_counts, growth_rate, n_classes, up_blocks, apply_softmax=False):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.up_blocks = up_blocks
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(self.up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(cur_channels_count, growth_rate, self.up_blocks[i], upsample=True))
            prev_block_channels = growth_rate * self.up_blocks[i]
            cur_channels_count += prev_block_channels

        self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]
        self.denseBlocksUp.append(DenseBlock(cur_channels_count, growth_rate, self.up_blocks[-1], upsample=False))
        cur_channels_count += growth_rate * self.up_blocks[-1]

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, out_channels=n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax2d()

    def forward(self, out, skip_connections):
        for i in range(len(self.up_blocks)):
            skip = skip_connections[-i - 1]
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        if self.apply_softmax:
            out = self.softmax(out)

        return out


class FCDenseNet3D(l.LightningModule):
    def __init__(self, hparams):
        super(FCDenseNet, self).__init__()
        #parameters
        self.up_blocks = hparams["up_blocks"]
        self.softmax = nn.Softmax2d()

        self.in_channels = hparams["in_channels"]
        self.down_blocks = hparams["down_blocks"]
        self.bottleneck_layers = hparams["bottleneck_layers"]
        self.growth_rate = hparams["growth_rate"]
        self.out_chans_first_conv = hparams["out_chans_first_conv"]
        self.n_classes = hparams["n_classes"]

        if self.n_classes == 4:
            assert hparams["remove_pleural_effusion"], "Remove pleural effusion function must be provided for 4 classes"
        elif self.n_classes == 5:
            assert hparams["remove_pleural_effusion"] == False, "Remove pleural effusion function must not be provided for 5 classes"
        else:
            raise ValueError("Number of classes must be 4 or 5")

        self.densenet_encoder = FCDenseNetEncoder(in_channels=self.in_channels, down_blocks=self.down_blocks,
                                                  bottleneck_layers=self.bottleneck_layers,
                                                  growth_rate=self.growth_rate, out_chans_first_conv=self.out_chans_first_conv)
        
        prev_block_channels = self.densenet_encoder.prev_block_channels
        skip_connection_channel_counts = self.densenet_encoder.skip_connection_channel_counts

        self.decoder = FCDenseNetDecoder(prev_block_channels, skip_connection_channel_counts, self.growth_rate, self.n_classes, self.up_blocks)


    def forward(self, x, device):
        out1, skip_connections = self.densenet_encoder(x.type(torch.FloatTensor).to(device))
        out = self.decoder(out1, skip_connections)
        return out
