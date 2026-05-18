"""
Adapt KPConvX to allow similar usage as pointcept defined.
Less flexible than original implementation but easier to follow...
"""

from functools import partial
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import pointops

from models.pointcept_structure import PointModule, PointSequential, Point
from models.sonata import Embedding, GridPooling, GridUnpooling
from models.kp_stuff.kp_blocks import KPConvD, KPConvX
from timm.models.layers import DropPath
from dataset.utils import dict_rename_for_pointcept


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels, momentum=0.05)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class KPNeXtBlock(nn.Module):
    """
    We do not want to make the NN calculations in this Block to avoid depth times calculations. To this end
    this block sticks with the tuple forward input and output...
    """

    def __init__(
        self,
        embed_channels,
        groups=0,
        drop_path_rate=0.0,
        radius=1.0,
        shell_sizes=[1, 14, 28],
        enable_checkpoint=False,
        influence_mode="linear",
        attention_act="sigmoid",
        mod_grp_norm=True,
    ):
        super(KPNeXtBlock, self).__init__()

        # Hard coded parameters
        sigma = radius
        shared_kp_data = None
        dimension = 3

        self.enable_checkpoint = enable_checkpoint

        # KPConvX or KPConvD
        if groups == 0:
            self.conv = KPConvD(
                embed_channels,
                shell_sizes,
                radius,
                sigma,
                Cmid=0,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
            )
        else:
            self.conv = KPConvX(
                embed_channels,
                shell_sizes,
                radius,
                sigma,
                attention_groups=groups,
                attention_act=attention_act,
                mod_grp_norm=mod_grp_norm,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
            )

        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))

        # feat = self.attn(feat, coord, reference_index) \
        #     if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index)

        # Here KPConvX
        feat = (
            self.conv(coord, coord, feat, reference_index)
            if not self.enable_checkpoint
            else checkpoint(self.conv, coord, coord, feat, reference_index)
        )

        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(PointModule):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        drop_path_rate=0.0,
        radius=1.0,
        shell_sizes=[1, 14, 28],
        enable_checkpoint=False,
        influence_mode="linear",
        attention_act="sigmoid",
        mod_grp_norm=True,
    ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = KPNeXtBlock(
                embed_channels=embed_channels,
                groups=groups,
                drop_path_rate=drop_path_rates[i],
                radius=radius,
                shell_sizes=shell_sizes,
                enable_checkpoint=enable_checkpoint,
                influence_mode=influence_mode,
                attention_act=attention_act,
                mod_grp_norm=mod_grp_norm,
            )
            self.blocks.append(block)

    def forward(self, point: Point):
        coord = point.coord
        feat = point.feat
        offset = point.offset
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)

        # Any index == -1 becomes the max index
        reference_index[reference_index == -1] = coord.shape[0]
        points_tuple = (coord, feat, offset)
        for block in self.blocks:
            points_tuple = block(points_tuple, reference_index)

        # update point structure with new feats
        point.feat = points_tuple[1]
        return point


class KPNeXt(nn.Module):
    def __init__(self, kp_config):
        super(KPNeXt, self).__init__()

        ############
        # Parameters
        ############
        self.in_channels = kp_config.in_channels
        self.nr_classes = kp_config.nr_classes

        drop_path_rate = kp_config.drop_path_rate
        enable_checkpoint = kp_config.enable_checkpoints

        stride = kp_config.stride
        enc_depths = kp_config.enc_depths
        enc_channels = kp_config.enc_channels
        enc_groups = kp_config.enc_groups
        enc_neighbours = kp_config.enc_neighbours
        dec_channels = kp_config.dec_channels
        influence_mode = kp_config.influence_mode
        attention_act = kp_config.attention_act

        # Parameters
        self.subsample_size = kp_config.voxel_size
        grid_sizes = [self.subsample_size]
        for s in stride:
            grid_sizes.append(grid_sizes[-1] * s)
        self.kp_radius = kp_config.radius
        self.first_radius = self.subsample_size * self.kp_radius
        self.shell_sizes = kp_config.shell_sizes

        # Get radii from grid sizes
        layer_radii = [dl * self.kp_radius for dl in grid_sizes]
        self.num_stages = len(enc_depths)

        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(grid_sizes)

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=self.in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        re_serialization=False,
                    ),
                    name="down",
                )
            enc.add(
                BlockSequence(
                    depth=enc_depths[s],
                    embed_channels=enc_channels[s],
                    groups=enc_groups[s],
                    neighbours=enc_neighbours[s],
                    drop_path_rate=enc_drop_path[
                        sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
                    ],
                    radius=layer_radii[s],
                    shell_sizes=self.shell_sizes,
                    enable_checkpoint=enable_checkpoint,
                    influence_mode=influence_mode,
                    attention_act=attention_act,
                ),
                name=f"block{s}",
            )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        self.dec = PointSequential()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec = PointSequential()
            dec.add(
                GridUnpooling(
                    in_channels=dec_channels[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),
                name="up",
            )
            self.dec.add(module=dec, name=f"dec{s}")

        self.seg_head = nn.Linear(dec_channels[0], self.nr_classes)

    def forward(self, data_dict):
        renamed_dict = dict_rename_for_pointcept(data_dict.copy())

        point = Point(renamed_dict)
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        out = self.seg_head(point.feat)
        return out
