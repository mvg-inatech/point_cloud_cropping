import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import kaiming_uniform_

import numpy as np

from models.litept import MLP
from .kp_points import load_kernels
from .kp_utils import index_select, gather, radius_gaussian


class UnaryBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module = nn.LeakyReLU(0.1),
    ):
        """
        Unary block with normalization and activation in pack mode.
        Args:
            in_channels (int): dimension input features
            out_channels (int): dimension input features
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(UnaryBlock, self).__init__()

        self.mlp = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = activation
        self.norm = nn.LayerNorm(out_channels)
        return

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class KPConvD(nn.Module):

    def __init__(
        self,
        channels: int,
        shell_sizes: list,
        radius: float,
        sigma: float,
        Cmid: int = 0,
        shared_kp_data=None,
        dimension: int = 3,
        influence_mode: str = "linear",
        fixed_kernel_points: str = "center",
        norm_type: str = "batch",
        bn_momentum: float = 0.1,
        activation: nn.Module = nn.LeakyReLU(0.1),
        inf: float = 1e6,
    ):
        """
        Mini KPConv. Basically a depthwise KPConv with nearest aggregation optimized to run faster.
        Option to use MLP instead of kernel to get neighbor weights (then we are similar to PointNeXt)
        Args:
            channels                     (int): The number of channels.
            shell_sizes                 (list): The number of kernel points per shell.
            radius                     (float): The radius used for kernel point init.
            sigma                      (float): The influence radius of each kernel point.
            Cmid                         (int): Dimension of mid f. 0 for depthwise conv, > 0 for PointConv style
            shared_kp_data              (None): Optional data dict shared across the layer
            dimension                  (int=3): The dimension of the point space.
            influence_mode      (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            norm_type            (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum           (float=0.10): Momentum for batch normalization
            activation              (nn.Module: Activation function. Use None for no activation.
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPConvD, self).__init__()

        # Save parameters
        self.channels = channels
        self.shell_sizes = shell_sizes
        self.K = int(np.sum(shell_sizes))
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf
        self.Cmid = Cmid

        # Initialize weights
        if Cmid > 0:
            self.weights = nn.Parameter(
                torch.zeros(size=(self.K, Cmid)), requires_grad=True
            )
            self.out_mlp = nn.Linear(Cmid * channels, channels)
        else:
            self.weights = nn.Parameter(
                torch.zeros(size=(self.K, channels)), requires_grad=True
            )

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.share_kp = shared_kp_data is not None
        self.first_kp = False

        if self.share_kp:
            self.first_kp = "k_pts" not in shared_kp_data
            self.shared_kp_data = shared_kp_data
            if self.first_kp:
                self.shared_kp_data["k_pts"] = self.initialize_kernel_points()
            self.register_buffer("kernel_points", self.shared_kp_data["k_pts"])
        else:
            self.shared_kp_data = {}
            kernel_points = self.initialize_kernel_points()
            self.register_buffer("kernel_points", kernel_points)
            self.shared_kp_data["k_pts"] = kernel_points

        # Merge and aggregation function
        # self.merge_op = torch.add
        self.merge_op = torch.mul
        # self.aggr_op = lambda x, dim=0: torch.max(x, dim=dim)[0]
        self.aggr_op = torch.sum

        # Handle mlp case
        if self.influence_mode == "mlp":
            if Cmid > 0:
                # MLP does not have it final linear layer. It is the same as point conv
                self.delta_mlp = UnaryBlock(
                    self.dimension, Cmid, norm_type, bn_momentum, activation
                )

            else:
                # Complete mlp
                self.delta_mlp = MLP(
                    in_channels=self.dimension,
                    hidden_channels=16,
                    out_channels=channels,
                    activation=activation,
                )

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=np.sqrt(5))
        return

    def initialize_kernel_points(self) -> Tensor:
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        kernel_points = load_kernels(
            self.radius,
            self.shell_sizes,
            dimension=self.dimension,
            fixed=self.fixed_kernel_points,
        )
        return torch.from_numpy(kernel_points).float()

    @torch.no_grad()
    def get_neighbors_influences(
        self, q_pts: Tensor, s_pts: Tensor, neighb_inds: Tensor
    ) -> Tensor:
        """
        Influence function of kernel points on neighbors.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        """

        if self.share_kp and not self.first_kp:

            # We use data already computed from the first KPConv of the layer
            influence_weights = self.shared_kp_data["infl_w"]
            neighbors = self.shared_kp_data["neighb_p"]
            neighbors_1nn = self.shared_kp_data["neighb_1nn"]

        else:

            # Add a fake point in the last row for shadow neighbors
            s_pts = torch.cat(
                (s_pts, torch.zeros_like(s_pts[:1, :]) + self.inf), 0
            )  # (N, 3) -> (N+1, 3)

            # Get neighbor points [n_points, n_neighbors, dim]
            # neighbors = s_pts[neighb_inds, :]  # (N+1, 3) -> (M, H, 3)
            neighbors = index_select(s_pts, neighb_inds, dim=0)  # (N+1, 3) -> (M, H, 3)

            # Center every neighborhood
            neighbors = neighbors - q_pts.unsqueeze(1)  # (M, H, 3)

            if self.influence_mode == "mlp":
                neighbors *= 1 / self.radius  # -> (M, H, 3)
                neighbors_1nn = None
                influence_weights = None

            else:

                # # Get Kernel point distances to neigbors
                # differences = (
                #     neighbors.unsqueeze(2) - self.kernel_points
                # )  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
                # sq_distances = torch.sum(differences**2, dim=3)  # (M, H, K)

                # use a bit less memory intensive way to compute sq_distances
                neighbors_sq = torch.sum(
                    neighbors**2, dim=-1, keepdim=True
                )  # (M, H, 1)
                kp_sq = torch.sum(self.kernel_points**2, dim=-1)  # (K,)
                dot = torch.einsum(
                    "mhd,kd->mhk", neighbors, self.kernel_points
                )  # (M, H, K)
                sq_distances = neighbors_sq + kp_sq - 2 * dot  # (M, H, K)
                sq_distances.clamp_(min=0)

                # Get nearest kernel point (M, H), values < K
                nn_sq_dists, neighbors_1nn = torch.min(sq_distances, dim=2)

                influence_weights = None
                if self.influence_mode != "constant":

                    # Get Kernel point influences
                    if self.influence_mode == "linear":
                        # Influence decrease linearly with the distance, and get to zero when d = sigma.
                        influence_weights = torch.clamp(
                            1 - torch.sqrt(nn_sq_dists) / self.sigma, min=0.0
                        )  # (M, H)

                    elif self.influence_mode == "gaussian":
                        # Influence in gaussian of the distance.
                        gaussian_sigma = self.sigma * 0.3
                        influence_weights = radius_gaussian(
                            nn_sq_dists, gaussian_sigma
                        )  # (M, H)
                    else:
                        raise ValueError(
                            "Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(
                                self.influence_mode
                            )
                        )

            # Share with next kernels if necessary
            if self.share_kp:

                self.shared_kp_data["neighb_1nn"] = neighbors_1nn
                self.shared_kp_data["neighb_p"] = neighbors
                self.shared_kp_data["infl_w"] = influence_weights

        return influence_weights, neighbors, neighbors_1nn

    def forward(
        self, q_pts: Tensor, s_pts: Tensor, s_feats: Tensor, neighb_inds: Tensor
    ) -> Tensor:
        """
        KPConv forward.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """

        # Add a zero feature for shadow neighbors
        padded_s_feats = torch.cat(
            (s_feats, torch.zeros_like(s_feats[:1, :])), 0
        )  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        # neighbor_feats = gather(padded_s_feats, neighb_inds)  # (N+1, C) -> (M, H, C)
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0)

        # Get nearest kernel point (M, H) and weights applied to each neighbors (M, H)
        influence_weights, neighbors, neighbors_1nn = self.get_neighbors_influences(
            q_pts, s_pts, neighb_inds
        )

        if self.influence_mode == "mlp":

            # Generate geometric encodings
            neighbors_weights = self.delta_mlp(neighbors)  # (M, H, 3) -> (M, H, C)

        else:

            # Collect nearest kernel point weights -> (M, H, C or Cmid)
            neighbors_weights = gather(self.weights, neighbors_1nn)

            # Apply influence weights
            if self.influence_mode != "constant":
                neighbors_weights *= influence_weights.unsqueeze(2)

        if self.Cmid > 0:

            # Apply weights via matmul
            intermediate_feats = torch.matmul(
                neighbors_weights.transpose(1, 2), neighbor_feats
            )  # (M, Cmid, H) x (M, H, C) -> (M, Cmid, C)

            # Final linear combination
            output_feats = self.out_mlp(
                intermediate_feats.view(-1, self.Cmid * self.channels)
            )

        else:

            # Apply weights and summation
            output_feats = self.aggr_op(
                self.merge_op(neighbor_feats, neighbors_weights), dim=1
            )  # (M, H, C) -> (M, C)

        return output_feats

    def __repr__(self):

        repr_str = "KPConvD"
        repr_str += "(K: {:d}".format(self.K)
        repr_str += ", C: {:d}".format(self.channels)
        repr_str += ", r: {:.2f}".format(self.radius)
        repr_str += ", sigma: {:.2f})".format(self.sigma)

        return repr_str


class KPConvX(nn.Module):

    def __init__(
        self,
        channels: int,
        shell_sizes: list,
        radius: float,
        sigma: float,
        attention_groups: int = 8,
        attention_act: str = "sigmoid",
        mod_grp_norm: bool = False,
        shared_kp_data=None,
        dimension: int = 3,
        influence_mode: str = "linear",
        fixed_kernel_points: str = "center",
        activation: nn.Module = nn.LeakyReLU(0.1),
        inf: float = 1e6,
    ):
        """
        Next generation KPConv. Basically a depthwise KPConv with nearest aggregation optimized to run faster,
        and self-generated modulation for attention.
        Args:
            channels                  (int): The number of input channels.
            shell_sizes                 (list): The number of kernel points per shell.
            radius                     (float): The radius used for kernel point init.
            sigma                      (float): The influence radius of each kernel point.
            attention_groups           (int=8): number of groups in attention (negative value for ch_per_grp).
            attention_act                (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            mod_grp_norm          (bool=False): Use group norm for modulations or not.
            shared_kp_data              (None): Optional data dict shared across the layer
            dimension                  (int=3): The dimension of the point space.
            influence_mode      (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            activation              (nn.Module: Activation function. Use None for no activation.
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPConvX, self).__init__()

        # Verification of group parameter
        if attention_groups > 0:
            assert (
                channels % attention_groups == 0
            ), "channels must be divisible by ch_per_grp."
            ch_per_grp = channels // attention_groups
        else:
            ch_per_grp = -attention_groups
            assert (
                channels % ch_per_grp == 0
            ), "channels must be divisible by ch_per_grp."
            attention_groups = channels // ch_per_grp

        # Save parameters
        self.channels = channels
        self.shell_sizes = shell_sizes
        self.K = int(np.sum(shell_sizes))
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf
        self.ch_per_grp = ch_per_grp
        self.groups = attention_groups
        self.attention_act = attention_act
        self.mod_grp_norm = mod_grp_norm

        # Depthwise conv parameters
        # *************************

        # Initialize weights
        self.weights = nn.Parameter(
            torch.zeros(size=(self.K, channels)), requires_grad=True
        )
        kaiming_uniform_(self.weights, a=math.sqrt(5))

        # Initialize kernel points
        self.share_kp = shared_kp_data is not None
        self.first_kp = False

        if self.share_kp:
            self.first_kp = "k_pts" not in shared_kp_data
            self.shared_kp_data = shared_kp_data
            if self.first_kp:
                self.shared_kp_data["k_pts"] = self.initialize_kernel_points()
            self.register_buffer("kernel_points", self.shared_kp_data["k_pts"])
        else:
            self.shared_kp_data = {}
            kernel_points = self.initialize_kernel_points()
            self.register_buffer("kernel_points", kernel_points)
            self.shared_kp_data["k_pts"] = kernel_points

        # Merge and aggregation function
        self.merge_op = torch.mul
        self.aggr_op = torch.sum

        # Attention parameters
        # ********************

        # Attention mlp
        Cout = self.K * self.ch_per_grp
        # alpha_list = [Cout]
        alpha_list = [channels, "NA", Cout]
        self.alpha_mlp = mlp_from_list(
            channels,
            alpha_list,
            final_bias=False,
            norm_type="none",
            bn_momentum=-1,
            activation=activation,
        )

        # Optional final group norm for each kernel weights
        self.grpnorm = nn.GroupNorm(self.K, self.K * self.ch_per_grp)
        # self.grpnorm = nn.BatchNorm1d(self.K * self.ch_per_grp, momentum=bn_momentum)

        # Weight activation
        if attention_act == "sigmoid":
            self.attention_act = torch.sigmoid
        elif attention_act == "tanh":
            self.attention_act = torch.tanh
        elif attention_act == "softmax":
            self.attention_act = nn.Softmax(dim=1)
        else:
            self.attention_act = nn.Identity()

        return

    def initialize_kernel_points(self) -> Tensor:
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        kernel_points = load_kernels(
            self.radius,
            self.shell_sizes,
            dimension=self.dimension,
            fixed=self.fixed_kernel_points,
        )
        return torch.from_numpy(kernel_points).float()

    @torch.no_grad()
    def get_neighbors_influences(
        self, q_pts: Tensor, s_pts: Tensor, neighb_inds: Tensor
    ) -> Tensor:
        """
        Influence function of kernel points on neighbors.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        """

        if self.share_kp and not self.first_kp:

            # We use data already computed from the first KPConv of the layer
            influence_weights = self.shared_kp_data["infl_w"]
            neighbors = self.shared_kp_data["neighb_p"]
            neighbors_1nn = self.shared_kp_data["neighb_1nn"]

        else:

            # Add a fake point in the last row for shadow neighbors
            s_pts = torch.cat(
                (s_pts, torch.zeros_like(s_pts[:1, :]) + self.inf), 0
            )  # (N, 3) -> (N+1, 3)

            # Get neighbor points [n_points, n_neighbors, dim]
            # neighbors = s_pts[neighb_inds, :]  # (N+1, 3) -> (M, H, 3)
            neighbors = index_select(s_pts, neighb_inds, dim=0)  # (N+1, 3) -> (M, H, 3)

            # Center every neighborhood
            neighbors = neighbors - q_pts.unsqueeze(1)  # (M, H, 3)

            # Get Kernel point distances to neigbors
            differences = (
                neighbors.unsqueeze(2) - self.kernel_points
            )  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
            sq_distances = torch.sum(differences**2, dim=3)  # (M, H, K)

            # Get nearest kernel point (M, H), values < K
            nn_sq_dists, neighbors_1nn = torch.min(sq_distances, dim=2)

            influence_weights = None
            if self.influence_mode != "constant":

                # Get Kernel point influences
                if self.influence_mode == "linear":
                    # Influence decrease linearly with the distance, and get to zero when d = sigma.
                    influence_weights = torch.clamp(
                        1 - torch.sqrt(nn_sq_dists) / self.sigma, min=0.0
                    )  # (M, H)

                elif self.influence_mode == "gaussian":
                    # Influence in gaussian of the distance.
                    gaussian_sigma = self.sigma * 0.3
                    influence_weights = radius_gaussian(
                        nn_sq_dists, gaussian_sigma
                    )  # (M, H)
                else:
                    raise ValueError(
                        "Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(
                            self.influence_mode
                        )
                    )

            # Share with next kernels if necessary
            if self.share_kp:

                self.shared_kp_data["neighb_1nn"] = neighbors_1nn
                self.shared_kp_data["neighb_p"] = neighbors
                self.shared_kp_data["infl_w"] = influence_weights

        return influence_weights, neighbors, neighbors_1nn

    def forward(
        self, q_pts: Tensor, s_pts: Tensor, s_feats: Tensor, neighb_inds: Tensor
    ) -> Tensor:
        """
        KPTransformer forward.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """

        # Get Neighbor features
        # *********************

        # Add a zero feature for shadow neighbors
        padded_s_feats = torch.cat(
            (s_feats, torch.zeros_like(s_feats[:1, :])), 0
        )  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        neighbor_feats = index_select(
            padded_s_feats, neighb_inds, dim=0
        )  # -> (M, H, C)

        # Get modulations
        # ***************

        # In case M == N, we can assume this is an in-place convolution.
        if q_pts.shape[0] == s_pts.shape[0]:
            pooled_feats = s_feats  # (M, C)
        else:
            pooled_feats = neighbor_feats[:, 0, :]  # nearest pool (M, H, C) -> (M, C)
            # pooled_feats = torch.max(neighbor_feats, dim=1)  # max pool (M, H, C) -> (M, C)
            # pooled_feats = torch.mean(neighbor_feats, dim=1)  # avg pool (M, H, C) -> (M, C)

        # MLP to get weights
        modulations = self.alpha_mlp(pooled_feats)  # (M, C) -> (M, C//r) -> (M, K*CpG)

        # Optional normalization per kernel
        if self.mod_grp_norm:
            modulations = modulations.transpose(0, 1).unsqueeze(
                0
            )  # (M, K*CpG) -> (B=1, K*CpG, M)
            modulations = self.grpnorm(modulations)
            modulations = modulations.squeeze(0).transpose(
                0, 1
            )  # (B=1, K*CpG, M) -> (M, K*CpG)

        # Activation
        modulations = self.attention_act(modulations)

        # Apply modulations
        # *****************

        # Reshapes
        modulations = modulations.view(
            -1, self.K, self.ch_per_grp, 1
        )  # -> (M, K, CpG, 1)
        conv_weights = self.weights.view(
            1, self.K, self.ch_per_grp, self.groups
        )  # -> (1, K, CpG, G)

        # Modulate convolution weights at each location (M, K, CpG, G)
        conv_weights = conv_weights * modulations

        # Reshape
        conv_weights = conv_weights.reshape(-1, self.K, self.channels)  # -> (M, K, C)

        # Depthwise convolution
        # *********************

        # Get nearest kernel point (M, H) and weights applied to each neighbors (M, H)
        influence_weights, neighbors, neighbors_1nn = self.get_neighbors_influences(
            q_pts, s_pts, neighb_inds
        )

        # Collect nearest kernel point weights (M, K, C) -> (M, H, C)
        neighbors_weights = torch.gather(
            conv_weights, 1, neighbors_1nn.unsqueeze(2).expand(-1, -1, self.channels)
        )

        # Adjust weights with influence
        if self.influence_mode != "constant":
            neighbors_weights *= influence_weights.unsqueeze(2)

        # Apply convolution weights
        neighbor_feats = self.merge_op(neighbor_feats, neighbors_weights)  # (M, H, C)

        # Output
        # ******

        # Final summation
        output_feats = self.aggr_op(neighbor_feats, dim=1)  # (M, H, C) -> (M, C)

        return output_feats

    def __repr__(self):

        repr_str = "KPConvX"
        repr_str += "(K: {:d}".format(self.K)
        repr_str += ", C: {:d}".format(self.channels)
        repr_str += ", r: {:.2f}".format(self.radius)
        repr_str += ", sigma: {:.2f})".format(self.sigma)

        return repr_str
