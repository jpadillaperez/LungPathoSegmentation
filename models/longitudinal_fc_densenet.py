from models.fc_densenet import FCDenseNetEncoder, FCDenseNetDecoder
from models.layers import *
#import pytorch_lightning as pl
import lightning as l
from utils.utils import get_argparser_group
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os

class LongitudinalFCDenseNet(l.LightningModule):
    def __init__(self, hparams, encoder=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.up_blocks = self.hparams.up_blocks
        self.densenet_encoder = encoder
        self.longitudinal = hparams.longitudinal
        self.siamese = self.hparams.siamese
        self.softmax = nn.Softmax2d()
        if not encoder:
            self.densenet_encoder = FCDenseNetEncoder(in_channels=self.hparams.in_channels * (1 if not self.longitudinal else 2), down_blocks=self.hparams.down_blocks,
                                                      bottleneck_layers=self.hparams.bottleneck_layers,
                                                      growth_rate=self.hparams.growth_rate, out_chans_first_conv=self.hparams.out_chans_first_conv)

        prev_block_channels = self.densenet_encoder.prev_block_channels
        skip_connection_channel_counts = self.densenet_encoder.skip_connection_channel_counts

        if self.siamese:
            self.merge_conv = nn.Conv2d(prev_block_channels * 2, prev_block_channels, 1, 1)

        self.decoder = FCDenseNetDecoder(prev_block_channels, skip_connection_channel_counts, self.hparams.growth_rate, self.hparams.n_classes, self.hparams.up_blocks)

    def forward(self, x_ref, x, cuda):
        if self.siamese:
            out, skip_connections = self.densenet_encoder(x)
            out_ref, _ = self.densenet_encoder(x_ref)
            out = torch.cat((out, out_ref), dim=1)
            out1 = self.merge_conv(out)
        else:
            if self.longitudinal:
                out1, skip_connections = self.densenet_encoder(torch.cat((x_ref, x), dim=1).type(torch.FloatTensor).to(cuda))
            else:
                out1, skip_connections = self.densenet_encoder(x.type(torch.FloatTensor).to(cuda))

        out = self.decoder(out1, skip_connections)

        return out


    @staticmethod
    def add_model_specific_args(parser):
        specific_args = get_argparser_group(title="Model options", parser=parser)
        specific_args.add_argument('--down_blocks', type=tuple, default=(4, 4, 4, 4, 4))
        specific_args.add_argument('--up_blocks', type=tuple, default=(4, 4, 4, 4, 4))
        specific_args.add_argument('--bottleneck_layers', type=int, default=4)
        specific_args.add_argument('--growth_rate', type=int, default=12)
        specific_args.add_argument('--out_chans_first_conv', type=int, default=48)
        specific_args.add_argument('--n_classes', type=int, default=4)
        specific_args.add_argument('--siamese', type=bool, default=False)
        return parser

    def test_step(self, batch: Any, batch_idx: int):
        x_ref = batch['image'][:, 0, :, :, :]
        x = batch['image'][:, 1, :, :, :]
        device = batch['image'].device

        y_pred = self.forward(x_ref, x, device)
        y_pred = self.softmax(y_pred)

        #print(y_pred.shape)

        output = y_pred[0, 0, :, :].cpu().detach().numpy() * 1 + y_pred[0, 1, :, :].cpu().detach().numpy() * 2 + y_pred[0, 2, :, :].cpu().detach().numpy() * 3 + y_pred[0, 3, :, :].cpu().detach().numpy() * 4

        #save image predicted
        plt.imshow(output)
        #print(self.hparams["output_path"])
        output_path = os.path.join(self.hparams["output_path"], "predicted" + str(batch_idx) + ".png")
        plt.savefig(output_path)
        plt.close()
        
        return {"y": y_pred}
    
