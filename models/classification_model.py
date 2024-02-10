from models.fc_densenet import FCDenseNetEncoder
from models.layers import *
import pytorch_lightning as pl
from utils.utils import get_argparser_group
from torch import nn

class LongitudinalClassification(pl.LightningModule):
    def __init__(self, hparams, encoder=None):
        super(LongitudinalClassification, self).__init__()
        self.save_hyperparameters(hparams)
        self.up_blocks = self.hparams.up_blocks
        self.densenet_encoder = encoder
        self.longitudinal = hparams.longitudinal
        self.siamese = self.hparams.siamese
        if not encoder:
            self.densenet_encoder = FCDenseNetEncoder(in_channels=self.hparams.in_channels * (1 if not self.longitudinal else 2), down_blocks=self.hparams.down_blocks,
                                                      bottleneck_layers=self.hparams.bottleneck_layers,
                                                      growth_rate=self.hparams.growth_rate, out_chans_first_conv=self.hparams.out_chans_first_conv)

        prev_block_channels = self.densenet_encoder.prev_block_channels

        if self.siamese:
            self.merge_conv = nn.Conv2d(prev_block_channels * 2, prev_block_channels, 1, 1)
        self.global_average_pooling = nn.AvgPool2d(kernel_size=8)
        self.linear_layer = nn.Linear(288, 2)




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

        out = self.global_average_pooling(out1)
        out = torch.squeeze(out, dim=3)
        out = torch.squeeze(out, dim=2)
        out = self.linear_layer(out)

        return out


    @staticmethod
    def add_model_specific_args(parser):
        specific_args = get_argparser_group(title="Model options", parser=parser)
        specific_args.add_argument('--down_blocks', type=tuple, default=(4, 4, 4, 4, 4))
        specific_args.add_argument('--up_blocks', type=tuple, default=(4, 4, 4, 4, 4))
        specific_args.add_argument('--bottleneck_layers', type=int, default=0)
        specific_args.add_argument('--growth_rate', type=int, default=12)
        specific_args.add_argument('--out_chans_first_conv', type=int, default=48)
        specific_args.add_argument('--siamese', type=bool, default=False)
        return parser
