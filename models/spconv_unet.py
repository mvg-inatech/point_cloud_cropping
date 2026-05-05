from functools import partial
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from timm.layers import trunc_normal_

from dataset.utils import dict_to_spconv


class DoubleConv(spconv.SparseModule):

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        self.proj = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_fn(out_channels),
        )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class DonwBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sp_indice_key,
        norm_fn=None,
        sub_indice_key=None,
    ):
        super().__init__()
        self.down = spconv.SparseSequential(
            spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=1,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(out_channels),
            nn.ReLU(),
        )
        self.enc = DoubleConv(out_channels, out_channels, norm_fn, sub_indice_key)

    def forward(self, x):
        x = self.down(x)
        x = self.enc(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        sp_indice_key,
        norm_fn=None,
        down_ratio=2,
    ):
        super().__init__()
        self.up = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                in_channels,
                skip_channels,
                kernel_size=down_ratio,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(skip_channels),
            nn.ReLU(),
        )
        self.fuse = DoubleConv(skip_channels * 2, out_channels, norm_fn)

    def forward(self, x, skip_x):
        x = self.up(x)
        x = x.replace_feature(torch.cat([x.features, skip_x.features], dim=1))
        x = self.fuse(x)
        return x


class SpConvUNet(nn.Module):
    def __init__(self, sp_config):
        super().__init__()
        in_channels = sp_config.in_channels
        self.nr_classes = sp_config.nr_classes
        enc_channels = sp_config.enc_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        expansion = 2
        self.in_channels = in_channels
        self.enc_channels = enc_channels

        self.stem = DoubleConv(in_channels, enc_channels[0], norm_fn)
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(enc_channels)):
            self.enc.append(
                DonwBlock(
                    in_channels=enc_channels[i],
                    out_channels=enc_channels[i] * expansion,
                    norm_fn=norm_fn,
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                )
            )
            self.dec.append(
                UpBlock(
                    in_channels=enc_channels[i] * expansion,
                    skip_channels=enc_channels[i],
                    out_channels=enc_channels[i],
                    norm_fn=norm_fn,
                    sp_indice_key=f"spconv{i}",
                )
            )

        self.final = spconv.SubMConv3d(enc_channels[0], self.nr_classes, kernel_size=1)
        self.apply(self._init_weights)

    def forward(self, input_dict):
        if isinstance(input_dict, dict):
            x = dict_to_spconv(input_dict, self.in_channels)
        elif isinstance(input_dict, spconv.SparseConvTensor):
            x = input_dict
        else:
            raise NotImplementedError

        x = self.stem(x)
        skips = [x]
        for i in range(len(self.enc_channels)):
            x = self.enc[i](x)
            skips.append(x)
        x = skips.pop(-1)
        for i in reversed(range(len(self.enc_channels))):
            skip = skips.pop(-1)
            x = self.dec[i](x, skip)
        x = self.final(x)
        return x.features

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
