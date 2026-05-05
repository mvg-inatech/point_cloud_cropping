from functools import partial
import sys
import torch
import torch.nn as nn
from timm.layers import DropPath
import spconv.pytorch as spconv
from timm.layers import trunc_normal_
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter
from dataset.utils import dict_to_spconv


class GroupedSubMConv3d(nn.Module):
    """
    Not cool... temp solution
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, **kwargs):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0

        if groups > 64:
            sys.exit(
                "Too many groups for GroupedSubMConv3d! - Performance may degrade."
            )

        self.groups = groups
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups

        # Create separate convolutions for each group
        self.convs = nn.ModuleList(
            [
                spconv.SubMConv3d(
                    self.in_channels_per_group,
                    self.out_channels_per_group,
                    kernel_size,
                    **kwargs,
                )
                for _ in range(groups)
            ]
        )

    def forward(self, x):
        # Split features by groups
        features = x.features
        group_features = torch.chunk(features, self.groups, dim=1)

        # Apply convolution to each group
        group_outputs = []
        for i, group_feat in enumerate(group_features):
            x_group = x.replace_feature(group_feat)
            out_group = self.convs[i](x_group)
            group_outputs.append(out_group.features)

        # Concatenate outputs
        out_features = torch.cat(group_outputs, dim=1)
        return x.replace_feature(out_features)


class BasicBlock(nn.Module):
    def __init__(
        self,
        embed_channels,
        norm_fn=None,
        indice_key=None,
        depth=4,
        groups=32,
        grid_size=None,
        bias=False,
        drop_path_rate=None,
    ):
        super().__init__()
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.weight = nn.ModuleList()
        self.l_w = nn.ModuleList()
        self.proj.append(
            nn.Sequential(
                nn.Linear(embed_channels, embed_channels, bias=False),
                norm_fn(embed_channels),
                nn.ReLU(),
            )
        )
        for _ in range(depth - 1):
            self.proj.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.l_w.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.weight.append(nn.Linear(embed_channels, embed_channels, bias=False))

        self.adaptive = nn.Linear(embed_channels, depth - 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Linear(embed_channels * 2, embed_channels, bias=False),
            norm_fn(embed_channels),
            nn.ReLU(),
        )

        self.groups = groups
        if self.groups == 0:
            # allow to load old models
            self.voxel_block = spconv.SparseSequential(
                spconv.SubMConv3d(
                    embed_channels,
                    embed_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    indice_key=indice_key,
                    bias=bias,
                ),
                norm_fn(embed_channels),
                nn.ReLU(),
                spconv.SubMConv3d(
                    embed_channels,
                    embed_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    indice_key=indice_key,
                    bias=bias,
                ),
                norm_fn(embed_channels),
            )
        else:
            self.voxel_1 = GroupedSubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
                indice_key=indice_key,
                bias=bias,
            )
            self.norm_1 = norm_fn(embed_channels)
            self.voxel_2 = GroupedSubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
                indice_key=indice_key,
                bias=bias,
            )
            self.norm_2 = norm_fn(embed_channels)
        self.act = nn.ReLU()

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, clusters):
        feat = x.features
        feats = []
        for i, cluster in enumerate(clusters):
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)
            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)
        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("l n, l n c -> l c", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.features
        feat = self.dropout(feat)
        res = feat
        x = x.replace_feature(feat)
        if self.groups == 0:
            x = self.voxel_block(x)
        else:
            x = self.voxel_1(x)
            x = x.replace_feature(self.norm_1(x.features))
            x = self.voxel_2(x)
            x = x.replace_feature(self.norm_2(x.features))
        x_feat = self.drop_path(x.features)
        x = x.replace_feature(self.act(x_feat + res))
        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        depth,
        sp_indice_key,
        point_grid_size,
        groups,
        norm_fn=None,
        sub_indice_key=None,
        drop_path_rate=None,
    ):
        super().__init__()

        if drop_path_rate is None:
            drop_path_rate = [0.0] * depth

        self.depth = depth
        self.point_grid_size = point_grid_size
        self.down = spconv.SparseSequential(
            spconv.SparseConv3d(
                in_channels,
                embed_channels,
                kernel_size=2,
                stride=2,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                BasicBlock(
                    embed_channels=embed_channels,
                    depth=len(point_grid_size) + 1,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                    groups=groups,
                    drop_path_rate=drop_path_rate[i],
                )
            )

    def forward(self, x):
        x = self.down(x)
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        clusters = []
        for grid_size in self.point_grid_size:
            cluster = voxel_grid(pos=coord, size=grid_size, batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)
        for block in self.blocks:
            x = block(x, clusters)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        sp_indice_key,
        norm_fn=None,
        down_ratio=2,
    ):
        super().__init__()
        self.up = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                in_channels,
                embed_channels,
                kernel_size=down_ratio,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(skip_channels + embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.ReLU(),
            nn.Linear(embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.ReLU(),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = x.replace_feature(
            self.fuse(torch.cat([x.features, skip_x.features], dim=1)) + x.features
        )
        return x


class OACNNs(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nr_classes = config.nr_classes
        in_channels = config.in_channels
        embed_channels = config.embed_channels
        enc_channels = config.enc_channels
        enc_depth = config.enc_depth
        dec_channels = config.dec_channels
        point_grid_size = config.point_grid_size

        # allow to load old models
        if hasattr(config, "groups"):
            groups = config.groups
        else:
            groups = [0] * len(enc_channels)
        if hasattr(config, "drop_path_rate"):
            drop_path_rate = config.drop_path_rate
        else:
            drop_path_rate = 0.0

        self.in_channels = in_channels
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        enc_dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depth))
        ]

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                DownBlock(
                    in_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    point_grid_size=point_grid_size[i],
                    groups=groups[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                    drop_path_rate=enc_dp_rates[
                        sum(enc_depth[:i]) : sum(enc_depth[: i + 1])
                    ],
                )
            )
            self.dec.append(
                UpBlock(
                    in_channels=(
                        enc_channels[-1]
                        if i == self.num_stages - 1
                        else dec_channels[i + 1]
                    ),
                    skip_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=dec_channels[i],
                    norm_fn=norm_fn,
                    sp_indice_key=f"spconv{i}",
                )
            )
        self.final = spconv.SubMConv3d(dec_channels[0], self.nr_classes, kernel_size=1)
        self.apply(self._init_weights)

    def forward(self, input_dict):
        if isinstance(input_dict, dict):
            x = dict_to_spconv(input_dict)
        elif isinstance(input_dict, spconv.SparseConvTensor):
            x = input_dict
        else:
            raise NotImplementedError

        x = self.stem(x)
        skips = [x]
        for i in range(self.num_stages):
            x = self.enc[i](x)
            skips.append(x)
        x = skips.pop(-1)
        for i in reversed(range(self.num_stages)):
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
