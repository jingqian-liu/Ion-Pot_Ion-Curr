import math
import torch
import torch.nn as nn
import json

from typing import List, Literal, Union, Optional, Dict
from dataclasses import dataclass, field, asdict
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from .nn.unet import BasicUNet
from .nn.pbgnn import (
    PBGNN,
    PBGNNConfig,
    PBGNNwithCrossInteraction,
    PBGNNwithBoundaryCrossInteraction,
)


def squared_hinge_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    """ref: https://www.tensorflow.org/api_docs/python/tf/keras/losses/SquaredHinge"""
    hinge_loss = torch.maximum(1 - y_true * y_pred, torch.zeros_like(y_true))
    squared_loss = torch.square(hinge_loss)
    return squared_loss


class SquaredHingeLoss(nn.Module):
    def __init__(self):
        super(SquaredHingeLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = squared_hinge_loss(y_pred, y_true)
        return loss.mean()


class PerceptronLoss(nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        true_sign = torch.where(y_true > self.threshold, 1.0, -1.0)
        pred_sign = y_pred - self.threshold
        sign_loss = torch.max(torch.zeros_like(pred_sign), -pred_sign * true_sign)
        return sign_loss.mean()


class MLSESModel(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super().__init__()
        self.layer1 = nn.Linear(dim1, dim2)
        self.layer2 = nn.Linear(dim2, dim3)
        self.layer3 = nn.Linear(dim3, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

    def load_ckpt(self, ckpt_file):
        ckpt = json.load(open(ckpt_file))

        with torch.no_grad():
            for i in range(1, 4):
                layer: nn.Linear = getattr(self, f"layer{i}")
                layer.weight.copy_(layer.weight.new_tensor(ckpt[f"W{i}"]))
                layer.bias.copy_(layer.bias.new_tensor(ckpt[f"b{i}"]))


class RefinedMLSESModel(nn.Module):
    def __init__(self, dim1, dim2, dim3, probe_radius):
        super().__init__()
        self.layer1 = nn.Linear(dim1, dim2)
        self.layer2 = nn.Linear(dim2, dim3)
        self.layer3 = nn.Linear(dim3, 1)

        self.probe_radius = probe_radius

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x


class SimpleConvMlsesModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))

    def forward(self, input_x):
        x = self.convs(input_x)

        # do prediction
        out = self.conv_out(x)
        return out


@dataclass(eq=False)
class MultiScaleConvMlsesModel(nn.Module):
    kernel_sizes: List[int] = field(default_factory=lambda: [1, 3, 5], hash=False)

    def __post_init__(self) -> None:
        super().__init__()
        if len(self.kernel_sizes) != 3:
            raise Exception("kernel_sizes must be a list of 3 elements")

        self.convs_1 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=self.kernel_sizes[0]),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=self.kernel_sizes[0]),
            nn.ReLU(),
        )

        self.convs_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=self.kernel_sizes[1], padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=self.kernel_sizes[1], padding="same"),
            nn.ReLU(),
        )

        self.convs_5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, input_x):
        x_1 = self.convs_1(input_x)
        x_3 = self.convs_3(x_1)
        x_5 = self.convs_5(x_3)

        # do prediction
        out = self.conv_out(torch.cat([x_1, x_3, x_5], dim=1))
        return out


@dataclass(eq=False)
class MultiScaleConv3dMlsesModel(nn.Module):
    kernel_sizes: List[int] = field(default_factory=lambda: [1, 3, 5], hash=False)

    def __post_init__(self) -> None:
        super().__init__()
        if len(self.kernel_sizes) != 3:
            raise Exception("kernel_sizes must be a list of 3 elements")

        self.convs_1 = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=self.kernel_sizes[0]),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=self.kernel_sizes[0]),
            nn.ReLU(),
        )

        self.convs_3 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=self.kernel_sizes[1], padding="same"),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=self.kernel_sizes[1], padding="same"),
            nn.ReLU(),
        )

        self.convs_5 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=self.kernel_sizes[2], padding="same"),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(nn.Conv3d(64, 1, kernel_size=1))

    def forward(self, input_x):
        x_1 = self.convs_1(input_x)
        x_3 = self.convs_3(x_1)
        x_5 = self.convs_5(x_3)

        # do prediction
        out = self.conv_out(torch.cat([x_1, x_3, x_5], dim=1))
        return out

    def encode(self, input_x):
        x_1 = self.convs_1(input_x)
        x_3 = self.convs_3(x_1)
        x_5 = self.convs_5(x_3)
        out = self.conv_out(torch.cat([x_1, x_3, x_5], dim=1))
        return torch.cat([x_1, x_3, x_5], dim=1), out


class FullyConvMlsesModel(nn.Module):
    def __init__(
        self, patch_size, input_dim, output_dim, model_dim=16, conv_depth=3
    ) -> None:
        super().__init__()
        assert patch_size // (2**conv_depth) > 1, (
            "Patch size mismatches with the conv_depth"
        )

        self.conv_depth = conv_depth
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(conv_depth):
            if i == 0:
                input_filter_num = input_dim
                output_filter_num = model_dim
            else:
                input_filter_num = model_dim // 2
                output_filter_num = model_dim

            # add module
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_filter_num,
                        output_filter_num,
                        kernel_size=3,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        output_filter_num,
                        output_filter_num,
                        kernel_size=3,
                        padding="same",
                    ),
                    nn.ReLU(),
                )
            )

            self.pools.append(nn.MaxPool2d(2))

            # grow dim
            model_dim *= 2

        self.conv_mid = nn.Sequential(
            nn.Conv2d(model_dim // 2, model_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
        )

        self.upconvs = nn.ModuleList()
        self.deconvs = nn.ModuleList()
        for i in range(conv_depth):
            input_filter_num = model_dim
            output_filter_num = model_dim // 2

            self.upconvs.append(
                nn.ConvTranspose2d(
                    input_filter_num, output_filter_num, kernel_size=2, stride=2
                )
            )
            self.deconvs.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_filter_num,
                        output_filter_num,
                        kernel_size=3,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        output_filter_num,
                        output_filter_num,
                        kernel_size=3,
                        padding="same",
                    ),
                    nn.ReLU(),
                )
            )

            # shrimp dim
            model_dim //= 2

        self.conv_out = nn.Sequential(nn.Conv2d(model_dim, output_dim, kernel_size=1))

    def forward(self, input_x):
        x = input_x
        conv_xs = []
        for i in range(self.conv_depth):
            x = self.convs[i](x)
            conv_xs.append(x)
            x = self.pools[i](x)

        x = self.conv_mid(x)
        deconv_xs = []
        for i in range(self.conv_depth):
            x = self.upconvs[i](x)
            x = torch.cat([x, conv_xs[-(i + 1)]], dim=1)
            x = self.deconvs[i](x)
            deconv_xs.append(x)

        # do prediction
        out = self.conv_out(x)
        return out


class SinusoidalPosEmb3D(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, None, None, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SinusoidalPosEmb1D(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GaussianRBFPosEmb(nn.Module):
    """Map scalar inputs to Gaussian RBF features on a bounded support."""

    def __init__(
        self,
        dim: int,
        value_min: float = -1.0,
        value_max: float = 1.0,
        gamma: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.gamma = gamma
        self.register_buffer("centers", torch.linspace(value_min, value_max, dim))

    def forward(self, x: torch.Tensor):
        # tanh keeps the dynamic range bounded for stable fixed-center RBF expansion.
        bounded_x = torch.tanh(x)
        return torch.exp(-self.gamma * (bounded_x[..., None] - self.centers) ** 2)


class ResnetBlock3d(nn.Module):
    """A simple residual block for 3D volumes."""

    def __init__(
        self,
        dim: int,
        padding_type: Literal["reflect", "replicate", "zero"],
        norm_layer,
        use_dropout: bool,
        use_bias: bool,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim,
            padding_type,
            norm_layer,
            use_dropout,
            use_bias,
            dropout_p,
        )

    def _conv_with_padding(
        self,
        dim: int,
        padding_type: Literal["reflect", "replicate", "zero"],
        use_bias: bool,
    ):
        layers = []
        padding = 0
        if padding_type == "reflect":
            layers.append(nn.ReflectionPad3d(1))
        elif padding_type == "replicate":
            layers.append(nn.ReplicationPad3d(1))
        elif padding_type == "zero":
            padding = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        layers.append(
            nn.Conv3d(dim, dim, kernel_size=3, padding=padding, bias=use_bias)
        )
        return layers

    def build_conv_block(
        self,
        dim: int,
        padding_type: Literal["reflect", "replicate", "zero"],
        norm_layer,
        use_dropout: bool,
        use_bias: bool,
        dropout_p: float,
    ):
        conv_block = []
        conv_block += self._conv_with_padding(dim, padding_type, use_bias)
        conv_block += [norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(dropout_p)]

        conv_block += self._conv_with_padding(dim, padding_type, use_bias)
        conv_block += [norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor):
        return x + self.conv_block(x)


class IonResnet3d(nn.Module):
    """A compact 3D ResNet encoder-decoder for voxel potential map prediction."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        use_dropout: bool = False,
        n_blocks: int = 6,
        dropout_p: float = 0.1,
        padding_type: Literal["reflect", "replicate", "zero"] = "reflect",
    ):
        super().__init__()

        # InstanceNorm is generally more stable than BatchNorm for small 3D batch sizes.
        norm_layer = lambda c: nn.InstanceNorm3d(c, affine=True)
        use_bias = True

        model = [
            nn.ReflectionPad3d(1),
            nn.Conv3d(
                input_nc, ngf, kernel_size=3, stride=1, padding=0, bias=use_bias
            ),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.Conv3d(
                ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias
            ),
            norm_layer(ngf * 2),
            nn.ReLU(True),
        ]

        for _ in range(n_blocks):
            model += [
                ResnetBlock3d(
                    dim=ngf * 2,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    dropout_p=dropout_p,
                )
            ]

        model += [
            nn.ConvTranspose3d(
                ngf * 2,
                ngf,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=use_bias,
            ),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.ReflectionPad3d(1),
            # Keep linear output. APBS potential maps include positive and negative values.
            nn.Conv3d(ngf, output_nc, kernel_size=3, padding=0, bias=True),
        ]
        self.model = nn.Sequential(*model)

    @staticmethod
    def _match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Center-crop or symmetric-pad to match reference spatial size."""
        td, th, tw = ref.shape[-3:]
        d, h, w = x.shape[-3:]

        if d > td:
            s = (d - td) // 2
            x = x[:, :, s : s + td, :, :]
        if h > th:
            s = (h - th) // 2
            x = x[:, :, :, s : s + th, :]
        if w > tw:
            s = (w - tw) // 2
            x = x[:, :, :, :, s : s + tw]

        pd = max(0, td - x.shape[-3])
        ph = max(0, th - x.shape[-2])
        pw = max(0, tw - x.shape[-1])
        if pd or ph or pw:
            pd0 = pd // 2
            ph0 = ph // 2
            pw0 = pw // 2
            x = F.pad(
                x,
                (pw0, pw - pw0, ph0, ph - ph0, pd0, pd - pd0),
                mode="constant",
                value=0.0,
            )
        return x

    def forward(self, input_x: torch.Tensor):
        out = self.model(input_x)
        if out.shape[-3:] != input_x.shape[-3:]:
            out = self._match_spatial(out, input_x)
        return out


class SpectralConv3d(nn.Module):
    """3D spectral convolution used by FNO."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                modes3,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def compl_mul3d(input_ft: torch.Tensor, weights: torch.Tensor):
        # (B, in_c, D, H, W) x (in_c, out_c, D, H, W) -> (B, out_c, D, H, W)
        return torch.einsum("bixyz,ioxyz->boxyz", input_ft, weights)

    def forward(self, x: torch.Tensor):
        batchsize, _, size_x, size_y, size_z = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        mx = min(self.modes1, size_x)
        my = min(self.modes2, size_y)
        mz = min(self.modes3, size_z // 2 + 1)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            size_x,
            size_y,
            size_z // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, :mx, :my, :mz] = self.compl_mul3d(
            x_ft[:, :, :mx, :my, :mz],
            self.weights1[:, :, :mx, :my, :mz],
        )
        out_ft[:, :, -mx:, :my, :mz] = self.compl_mul3d(
            x_ft[:, :, -mx:, :my, :mz],
            self.weights2[:, :, :mx, :my, :mz],
        )
        out_ft[:, :, :mx, -my:, :mz] = self.compl_mul3d(
            x_ft[:, :, :mx, -my:, :mz],
            self.weights3[:, :, :mx, :my, :mz],
        )
        out_ft[:, :, -mx:, -my:, :mz] = self.compl_mul3d(
            x_ft[:, :, -mx:, -my:, :mz],
            self.weights4[:, :, :mx, :my, :mz],
        )

        x = torch.fft.irfftn(out_ft, s=(size_x, size_y, size_z), dim=(-3, -2, -1))
        return x


class FNO3d(nn.Module):
    """Compact 3D FNO mapper for voxel potential map prediction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        width: int = 32,
        depth: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        modes3: int = 12,
        padding: int = 6,
        use_grid_coords: bool = True,
    ) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        self.padding = padding
        self.use_grid_coords = use_grid_coords

        input_dim = in_channels + (3 if use_grid_coords else 0)
        self.input_proj = nn.Conv3d(input_dim, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [
                SpectralConv3d(width, width, modes1, modes2, modes3)
                for _ in range(depth)
            ]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv3d(width, width, kernel_size=1) for _ in range(depth)]
        )
        self.output_proj = nn.Sequential(
            nn.Conv3d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width * 2, out_channels, kernel_size=1),
        )

    @staticmethod
    def get_grid(shape: torch.Size, device: torch.device):
        batchsize, _, size_x, size_y, size_z = shape
        gridx = torch.linspace(0, 1, size_x, device=device).view(1, 1, size_x, 1, 1)
        gridy = torch.linspace(0, 1, size_y, device=device).view(1, 1, 1, size_y, 1)
        gridz = torch.linspace(0, 1, size_z, device=device).view(1, 1, 1, 1, size_z)
        gridx = gridx.expand(batchsize, 1, size_x, size_y, size_z)
        gridy = gridy.expand(batchsize, 1, size_x, size_y, size_z)
        gridz = gridz.expand(batchsize, 1, size_x, size_y, size_z)
        return torch.cat([gridx, gridy, gridz], dim=1)

    def forward(self, x: torch.Tensor):
        if self.use_grid_coords:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat([x, grid], dim=1)

        x = self.input_proj(x)
        if self.padding > 0:
            x = F.pad(
                x,
                (0, self.padding, 0, self.padding, 0, self.padding),
                mode="constant",
                value=0.0,
            )

        for i in range(self.depth):
            x1 = self.spectral_layers[i](x)
            x2 = self.pointwise_layers[i](x)
            x = F.gelu(x1 + x2)

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding, : -self.padding]

        return self.output_proj(x)


@dataclass
class MultiScaleConv3dEPBModelConfig:
    hidden_dim: int = 16
    atom_type_num: int = 8
    patch_size: int = 64
    zoom_patch_size: Union[None, int] = None
    reaction_size: int = 8
    kernel_sizes: List[int] = field(default_factory=lambda: [1, 3, 5], hash=False)
    embed_dim_scale: int = 4
    use_avg_pool: bool = False
    pooling_scale: int = 4
    lvlset_embed_scale: float = 10
    lvlset_embed_theta: float = 10
    charge_embed_scale: float = 1e2
    charge_embed_theta: float = 1e-1
    dropout_rate: float = 0.5
    block_drop_rate: float = 0.5
    use_lset: bool = True
    use_atom_num: bool = False
    is_atom_wise_potential_trained: bool = False
    output_kernel_size: int = 1
    reaction_field_mapping_version: Literal["u-shape", "u-net", "fno", "resnet"] = (
        "u-shape"
    )
    use_charge_diffusion: bool = True
    use_ionic_conc: bool = False
    ionic_conc_embed_scale: float = 10.0
    ionic_conc_embed_theta: float = 10000.0
    embedding_mode: Literal["sinusoidal", "rbf", "mlp", "hybrid"] = "sinusoidal"
    ionic_conditioning: Literal["add", "film"] = "add"
    feature_repr_strategy: Literal["emb", "concat"] = "emb"
    use_grid_coord: bool = False
    resnet_ngf: int = 64
    resnet_n_blocks: int = 6
    resnet_use_dropout: bool = False
    resnet_dropout_p: float = 0.1
    resnet_padding_type: Literal["reflect", "replicate", "zero"] = "reflect"
    fno_width: int = 32
    fno_depth: int = 4
    fno_modes1: int = 12
    fno_modes2: int = 12
    fno_modes3: int = 12
    fno_padding: int = 6
    fno_use_grid_coords: bool = True


@dataclass(eq=False)
class MultiScaleConv3dEPBModel(nn.Module):
    config: MultiScaleConv3dEPBModelConfig

    def __post_init__(self) -> None:
        super().__init__()
        # set config
        for k, v in asdict(self.config).items():
            setattr(self, k, v)

        # set zoom in patch slice
        self.zoom_slice = (
            slice(
                (self.patch_size - self.zoom_patch_size) // 2,
                self.zoom_patch_size + (self.patch_size - self.zoom_patch_size) // 2,
            )
            if self.zoom_patch_size is not None
            else slice(None)
        )

        # embeddings
        self.__init_embeddings()

        self.__init_diffusion()

        # reaction field mappings
        self.__init_reaction_field_mapping()

    def _build_scalar_embedding_3d(self, theta: float):
        mode = self.config.embedding_mode
        if mode in {"sinusoidal", "hybrid"}:
            return nn.Sequential(
                SinusoidalPosEmb3D(self.hidden_dim * self.embed_dim_scale, theta),
                nn.Linear(self.hidden_dim * self.embed_dim_scale, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        if mode == "rbf":
            return nn.Sequential(
                GaussianRBFPosEmb(self.hidden_dim * self.embed_dim_scale),
                nn.Linear(self.hidden_dim * self.embed_dim_scale, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        if mode == "mlp":
            return nn.Sequential(
                nn.Conv3d(1, self.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            )
        raise ValueError(f"Invalid embedding mode: {mode}")

    def _embed_scalar_3d(
        self,
        x: torch.Tensor,
        embedding: nn.Module,
        mode: Literal["sinusoidal", "rbf", "mlp", "hybrid"],
        projector: Optional[nn.Module] = None,
    ):
        if mode in {"sinusoidal", "rbf"}:
            return embedding(x.squeeze(1)).permute(0, 4, 1, 2, 3)
        if mode == "mlp":
            return embedding(x)
        if mode == "hybrid":
            emb = embedding(x.squeeze(1)).permute(0, 4, 1, 2, 3)
            if projector is None:
                raise ValueError("Projector is required for hybrid embedding mode.")
            return projector(torch.cat([emb, x], dim=1))
        raise ValueError(f"Invalid embedding mode: {mode}")

    def _apply_ionic_conditioning(self, x: torch.Tensor, ionic_conc: torch.Tensor):
        conc_emb = self.ionic_conc_embedding(ionic_conc * self.ionic_conc_embed_scale)
        if self.config.ionic_conditioning == "add":
            conc_emb = conc_emb[:, :, None, None, None].expand_as(x)
            return x + conc_emb
        if self.config.ionic_conditioning == "film":
            gamma, beta = torch.chunk(conc_emb, 2, dim=1)
            gamma = gamma[:, :, None, None, None]
            beta = beta[:, :, None, None, None]
            return x * (1.0 + gamma) + beta
        raise ValueError(
            f"Invalid ionic conditioning mode: {self.config.ionic_conditioning}"
        )

    def __init_embeddings(self):
        if self.config.feature_repr_strategy == "emb":
            # embeddings
            if self.use_lset:
                self.lset_embedding = self._build_scalar_embedding_3d(
                    self.lvlset_embed_theta
                )
                if self.config.embedding_mode == "hybrid":
                    self.lset_hybrid_projector = nn.Sequential(
                        nn.Conv3d(self.hidden_dim + 1, self.hidden_dim, kernel_size=1),
                        nn.GELU(),
                        nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=1),
                    )

            if self.use_atom_num:
                self.anum_embedding = nn.Sequential(
                    SinusoidalPosEmb3D(self.hidden_dim * self.embed_dim_scale),
                    nn.Linear(self.hidden_dim * self.embed_dim_scale, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )  # for each grid

            # self.atom_type_embedding = nn.Sequential(
            #     nn.Embedding(self.atom_type_num, self.hidden_dim * self.embed_dim_scale),
            #     nn.Linear(self.hidden_dim * self.embed_dim_scale, self.hidden_dim),
            #     nn.GELU(),
            #     nn.Linear(self.hidden_dim, self.hidden_dim),
            # )  # for each atom
            self.atom_charge_embedding = self._build_scalar_embedding_3d(
                self.charge_embed_theta
            )
            if self.config.embedding_mode == "hybrid":
                self.charge_hybrid_projector = nn.Sequential(
                    nn.Conv3d(self.hidden_dim + 1, self.hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=1),
                )

            if self.use_ionic_conc:
                ionic_out_dim = (
                    self.hidden_dim
                    if self.config.ionic_conditioning == "add"
                    else self.hidden_dim * 2
                )
                self.ionic_conc_embedding = nn.Sequential(
                    SinusoidalPosEmb1D(
                        self.hidden_dim * self.embed_dim_scale,
                        self.ionic_conc_embed_theta,
                    ),
                    nn.Linear(self.hidden_dim * self.embed_dim_scale, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, ionic_out_dim),
                )

    def __init_diffusion(self):
        if self.use_charge_diffusion:
            if self.reaction_size % 2 != 1:
                print("reaction_size must be odd")
                self.reaction_size += 1

            self.register_buffer(
                "reaction_field_range",
                (
                    torch.arange(
                        -self.reaction_size // 2 + 1, self.reaction_size // 2 + 1
                    )
                ),
            )

            # self.reaction_field_dropout = DropBlock3d(
            #     self.config.block_drop_rate, self.reaction_size
            # )
            self.reaction_field_dropout = nn.Identity()

            # parameterize the diagonal covariance of the diffusion process
            self.diffusion_sigma = nn.Parameter(torch.ones((3,)))

    def __init_reaction_field_mapping(self):
        version = self.config.reaction_field_mapping_version
        if version == "u-shape":
            self.__init_reaction_field_mapping_u_shape()
        elif version == "u-net":
            self.__init_reaction_field_mapping_unet()
        elif version == "fno":
            self.__init_reaction_field_mapping_fno()
        elif version == "resnet":
            self.__init_reaction_field_mapping_resnet()
        else:
            raise ValueError("Invalid reaction field mapping version")

    def __init_reaction_field_mapping_u_shape(self):
        # reaction field mappings
        self.reaction_field_mapping_en1 = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                self.hidden_dim,
                kernel_size=self.kernel_sizes[1],
                padding="same",
            ),
            nn.GELU(),
            # DropBlock3d(p=self.config.block_drop_rate, block_size=3),
        )

        self.reaction_field_mapping_en2 = nn.Sequential(
            (
                nn.AvgPool3d(self.pooling_scale)
                if self.use_avg_pool
                else nn.MaxPool3d(self.pooling_scale)
            ),
            nn.Conv3d(
                self.hidden_dim,
                self.hidden_dim * 2,
                kernel_size=self.kernel_sizes[2],
                padding="same",
            ),
            nn.GELU(),
            # DropBlock3d(p=self.config.block_drop_rate, block_size=1),
        )

        self.reaction_field_mapping_en3 = nn.Sequential(
            (
                nn.AvgPool3d(self.pooling_scale)
                if self.use_avg_pool
                else nn.MaxPool3d(self.pooling_scale)
            ),
            nn.Conv3d(
                self.hidden_dim * 2,
                self.hidden_dim * 4,
                kernel_size=self.kernel_sizes[1],
                padding="same",
            ),
            nn.GELU(),
            nn.Upsample(scale_factor=self.pooling_scale),
            # DropBlock3d(p=self.config.block_drop_rate, block_size=1),
        )

        self.reaction_field_mapping_de1 = nn.Sequential(
            nn.Conv3d(
                self.hidden_dim * 6,
                self.hidden_dim * 3,
                kernel_size=self.kernel_sizes[1],
                padding="same",
            ),
            nn.GELU(),
            nn.Upsample(scale_factor=self.pooling_scale),
        )

        self.reaction_field_mapping_de2 = nn.Sequential(
            nn.Conv3d(
                self.hidden_dim * 4,
                self.hidden_dim * 2,
                kernel_size=self.kernel_sizes[1],
                padding="same",
            ),
            nn.GELU(),
            nn.Conv3d(
                self.hidden_dim * 2,
                1,
                kernel_size=self.output_kernel_size,
                padding="same",
            ),
        )

    def __init_reaction_field_mapping_unet(self):
        h1 = self.hidden_dim
        h2 = self.hidden_dim * 2
        h4 = self.hidden_dim * 4
        features = (h1, h1, h2, h2, h4, h1)
        self.reaction_field_mapping_model = BasicUNet(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=1,
            features=features,
            output_kernel_size=self.output_kernel_size,
        )

    def __init_reaction_field_mapping_resnet(self):
        self.reaction_field_mapping_model = IonResnet3d(
            input_nc=self.in_channels,
            output_nc=1,
            ngf=self.config.resnet_ngf,
            n_blocks=self.config.resnet_n_blocks,
            use_dropout=self.config.resnet_use_dropout,
            dropout_p=self.config.resnet_dropout_p,
            padding_type=self.config.resnet_padding_type,
        )

    def __init_reaction_field_mapping_fno(self):
        self.reaction_field_mapping_model = FNO3d(
            in_channels=self.in_channels,
            out_channels=1,
            width=self.config.fno_width,
            depth=self.config.fno_depth,
            modes1=self.config.fno_modes1,
            modes2=self.config.fno_modes2,
            modes3=self.config.fno_modes3,
            padding=self.config.fno_padding,
            use_grid_coords=self.config.fno_use_grid_coords,
        )

    @property
    def in_channels(self):
        if self.config.feature_repr_strategy == "emb":
            return self.hidden_dim
        elif self.config.feature_repr_strategy == "concat":
            h = 1
            if self.config.use_lset:
                h += 1
            if self.config.use_atom_num:
                h += 1
            if self.config.use_charge_diffusion:
                h += 1
            if self.config.use_grid_coord:
                h += 3
            if self.config.use_ionic_conc:
                h += 1
            return h
        else:
            raise ValueError(
                f"Invalid feature representation strategy: {self.config.feature_repr_strategy}"
            )

    @staticmethod
    def split_grid_info(grid_info: torch.Tensor):
        """Support both legacy scalar grid_space [B] and new [B,2]=[grid_space, ionic_conc]."""
        if grid_info.ndim == 0:
            return grid_info.unsqueeze(0), None
        if grid_info.ndim == 1:
            return grid_info, None
        grid_space = grid_info[..., 0]
        ionic_conc = grid_info[..., 1] if grid_info.shape[-1] > 1 else None
        return grid_space, ionic_conc

    def init_gird_space_filter(self, unique_grid_space: torch.Tensor):
        reaction_field_filter = torch.meshgrid(
            self.reaction_field_range,
            self.reaction_field_range,
            self.reaction_field_range,
            indexing="ij",
        )

        reaction_field_filter = (
            torch.stack(reaction_field_filter, dim=-1).unsqueeze_(0)
            * unique_grid_space[..., None, None, None, None]
        )  # [grid_num, reaction_sz, reaction_sz, reaction_sz, 3]

        # create the learned gaussian distribution
        device = unique_grid_space.device
        gaussian = MultivariateNormal(
            torch.zeros(3, device=device), torch.diag(F.softplus(self.diffusion_sigma))
        )

        # compute the decay ratio
        diffusion_kernel = gaussian.log_prob(reaction_field_filter).exp()
        diffusion_kernel = diffusion_kernel / diffusion_kernel.sum(
            dim=(1, 2, 3), keepdim=True
        )

        # [grid_num, 1, reaction_sz, reaction_sz, reaction_sz]
        return diffusion_kernel.unsqueeze(1)

    def diffusion_charge(
        self,
        atom_charge: torch.Tensor,
        atom_mask: torch.Tensor,
        grid_space: torch.Tensor,
    ):
        unique_grid_space, inverse_indices = torch.unique(
            grid_space, sorted=True, return_inverse=True
        )
        diffusion_kernel = self.init_gird_space_filter(unique_grid_space)

        gather_indices = inverse_indices[:, None, None, None, None].expand(
            -1, -1, self.patch_size, self.patch_size, self.patch_size
        )
        diffused_charge = F.conv3d(
            atom_charge,
            diffusion_kernel,
            padding="same",
            groups=1,
        )
        diffused_charge = self.reaction_field_dropout(
            torch.gather(diffused_charge, 1, gather_indices)
        )

        diffused_charge = F.conv3d(
            diffused_charge,
            diffusion_kernel,
            padding="same",
            groups=1,
        )
        diffused_charge = self.reaction_field_dropout(
            torch.gather(diffused_charge, 1, gather_indices)
        )

        return diffused_charge

    def atom_embedding(
        self,
        atom_charge: torch.Tensor,
        atom_type: torch.Tensor,
        atom_mask: torch.Tensor,
        grid_space: torch.Tensor,
    ):
        charge_input = atom_charge.clone().detach()
        if self.use_charge_diffusion:
            diffused_charge = self.diffusion_charge(atom_charge, atom_mask, grid_space)
            charge_input = diffused_charge + charge_input
        scaled_charge = charge_input * self.charge_embed_scale
        x = self._embed_scalar_3d(
            scaled_charge,
            self.atom_charge_embedding,
            self.config.embedding_mode,
            projector=getattr(self, "charge_hybrid_projector", None),
        )

        # atype = self.atom_type_embedding(atom_type.squeeze(1).long()).permute(
        #     0, 4, 1, 2, 3
        # )
        # return charge + atype

        if self.use_atom_num:
            atom_num_emb = self.anum_embedding(atom_mask.squeeze(1)).permute(
                0, 4, 1, 2, 3
            )
            x += atom_num_emb
        return x

    def level_set_embedding(self, level_set: torch.Tensor):
        scaled_level_set = level_set * self.lvlset_embed_scale
        return self._embed_scalar_3d(
            scaled_level_set,
            self.lset_embedding,
            self.config.embedding_mode,
            projector=getattr(self, "lset_hybrid_projector", None),
        )

    def encode(
        self,
        voxel_level_set: torch.Tensor,
        atom_charge: torch.Tensor,
        atom_type: torch.Tensor,
        atom_mask: torch.Tensor,
        grid_space: torch.Tensor,
    ):
        grid_space, ionic_conc = self.split_grid_info(grid_space)
        if self.config.feature_repr_strategy == "emb":
            atom_emb = self.atom_embedding(
                atom_charge, atom_type, atom_mask, grid_space
            )
            # lset_emb = self.level_set_embedding(voxel_level_set)

            # compute reaction field potential
            # x = atom_emb + lset_emb

            x = atom_emb
            if self.use_lset:
                lset_emb = self.level_set_embedding(voxel_level_set)
                x += lset_emb
            if self.use_ionic_conc:
                if ionic_conc is None:
                    ionic_conc = torch.zeros_like(grid_space)
                x = self._apply_ionic_conditioning(x, ionic_conc)
        elif self.config.feature_repr_strategy == "concat":
            features = [
                atom_charge * self.charge_embed_scale,
            ]

            if self.config.use_charge_diffusion:
                diffused_charge = self.diffusion_charge(
                    atom_charge, atom_mask, grid_space
                )
                features.append(diffused_charge * self.charge_embed_scale)

            if self.config.use_lset:
                features.append(voxel_level_set * self.lvlset_embed_scale)

            if self.config.use_atom_num:
                features.append(atom_mask)

            if self.config.use_grid_coord:
                grid = self.get_grid_3d(atom_charge.shape, atom_charge.device)
                features.append(grid)

            if self.config.use_ionic_conc:
                if ionic_conc is None:
                    ionic_conc = torch.zeros_like(grid_space)
                conc_grid = ionic_conc[:, None, None, None, None].expand(
                    -1,
                    1,
                    atom_charge.shape[2],
                    atom_charge.shape[3],
                    atom_charge.shape[4],
                )
                features.append(conc_grid * self.ionic_conc_embed_scale)

            x = torch.cat(
                features,
                dim=1,
            )
        else:
            raise ValueError(
                f"Invalid feature representation strategy: {self.config.feature_repr_strategy}"
            )
        return x

    def reaction_field_mapping_u_shape(self, rf: torch.Tensor):
        x1 = self.reaction_field_mapping_en1(rf)
        x2 = self.reaction_field_mapping_en2(x1)  # 16
        x3 = self.reaction_field_mapping_en3(x2)  # 16

        dx2 = self.reaction_field_mapping_de1(torch.cat([x3, x2], dim=1))  # 64
        dx1 = self.reaction_field_mapping_de2(torch.cat([dx2, x1], dim=1))  # 64
        return dx1

    def reaction_field_mapping_u_net(self, rf: torch.Tensor):
        logits, _ = self.reaction_field_mapping_model(rf)
        return logits

    def reaction_field_mapping_fno(self, rf: torch.Tensor):
        logits = self.reaction_field_mapping_model(rf)
        return logits

    def reaction_field_mapping_resnet(self, rf: torch.Tensor):
        logits = self.reaction_field_mapping_model(rf)
        return logits

    def reaction_field_mapping(self, rf: torch.Tensor):
        version = self.config.reaction_field_mapping_version
        if version == "u-shape":
            return self.reaction_field_mapping_u_shape(rf)
        elif version == "u-net":
            return self.reaction_field_mapping_u_net(rf)
        elif version == "fno":
            return self.reaction_field_mapping_fno(rf)
        elif version == "resnet":
            return self.reaction_field_mapping_resnet(rf)
        else:
            raise ValueError("Invalid reaction field mapping version")

    def get_zoom_tensor(self, tensor):
        return tensor[..., self.zoom_slice, self.zoom_slice, self.zoom_slice]

    def compute_patch_epb(self, atom_charge, atom_mask, atom_potential):
        atom_mask = self.get_zoom_tensor(atom_mask)
        atom_charge = self.get_zoom_tensor(atom_charge)
        atom_potential = self.get_zoom_tensor(atom_potential)
        atom_bool_mask = atom_mask > 0

        epb = (
            torch.sum(
                torch.where(
                    atom_bool_mask,
                    atom_charge * atom_potential,
                    0.0,
                ),
                dim=(4, 3, 2),
            )
            * 0.5
        )
        return epb

    def get_grid_3d(self, shape, device):
        # shape in NCDHW
        batchsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]

        # Generate 1D grids for each dimension
        gridx = torch.linspace(0, 1, size_x, device=device)
        gridy = torch.linspace(0, 1, size_y, device=device)
        gridz = torch.linspace(0, 1, size_z, device=device)

        # Create 3D meshgrid
        gridx, gridy, gridz = torch.meshgrid(gridx, gridy, gridz, indexing="ij")

        # Reshape and repeat the grids for the batch dimension
        gridx = gridx.unsqueeze(0).repeat(
            batchsize, 1, 1, 1
        )  # Shape: (batchsize, size_x, size_y, size_z)
        gridy = gridy.unsqueeze(0).repeat(
            batchsize, 1, 1, 1
        )  # Shape: (batchsize, size_x, size_y, size_z)
        gridz = gridz.unsqueeze(0).repeat(
            batchsize, 1, 1, 1
        )  # Shape: (batchsize, size_x, size_y, size_z)

        # Stack the 3D grids along the last dimension to create the full grid
        grid = torch.stack(
            (gridx, gridy, gridz), dim=-1
        )  # Shape: (batchsize, size_x, size_y, size_z, 3)

        grid = grid.permute(0, 4, 1, 2, 3)
        return grid

    def forward(
        self,
        voxel_level_set: torch.Tensor,
        atom_charge: torch.Tensor,
        atom_type: torch.Tensor,
        atom_mask: torch.Tensor,
        grid_space: torch.Tensor,
        return_per_epb: bool = False,
    ):
        """for one voxel data

        :param voxel_x: P, C, D, H, W
        :return: P, D, H, W
        """
        rf = self.encode(voxel_level_set, atom_charge, atom_type, atom_mask, grid_space)
        per_epb = self.reaction_field_mapping(rf)

        # predict epb
        if self.is_atom_wise_potential_trained:
            epb = self.compute_patch_epb(atom_charge, atom_mask, per_epb)
        else:
            zoomed_per_epb = self.get_zoom_tensor(per_epb)
            zoomed_atom_charge = self.get_zoom_tensor(atom_charge)
            zoomed_atom_mask = self.get_zoom_tensor(atom_mask)
            zoomed_atom_bool_mask = zoomed_atom_mask > 0
            epb = torch.sum(
                -zoomed_per_epb * zoomed_atom_charge * zoomed_atom_bool_mask.float(),
                dim=(4, 3, 2),
            )

        outputs = (epb,)
        if return_per_epb:
            outputs += (per_epb,)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


@dataclass
class MultiScaleAtomicEPBModelConfig:
    hidden_dim: int = 16
    atom_type_num: int = 8
    embed_dim_scale: int = 4
    lvlset_embed_scale: float = 10
    lvlset_embed_theta: float = 10000
    charge_embed_scale: float = 1e2
    charge_embed_theta: float = 10000
    use_lset: bool = True
    is_atom_wise_potential_trained: bool = False
    reaction_field_mapping_version: Literal[
        "pbgnn", "pbgnn_fusion", "pbgnn_bndy_fusion"
    ] = "pbgnn"
    feature_repr_strategy: Literal["emb"] = "emb"
    pbgnn_config: PBGNNConfig = field(default_factory=PBGNNConfig)

    # shallow embedding (only SinusoidalPosEmb1D)
    use_shallow_embedding: bool = False

    # use boundary condition
    use_boundary_charge: bool = False
    use_boundary_lset: bool = False


class MultiScaleAtomicEPBModel(nn.Module):
    def __init__(self, config: MultiScaleAtomicEPBModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.embed_dim_scale = config.embed_dim_scale
        self.lvlset_embed_scale = config.lvlset_embed_scale
        self.lvlset_embed_theta = config.lvlset_embed_theta
        self.charge_embed_scale = config.charge_embed_scale
        self.charge_embed_theta = config.charge_embed_theta
        self.is_atom_wise_potential_trained = config.is_atom_wise_potential_trained

        # embeddings
        self.__init_embeddings()

        # reaction field mappings
        self.__init_reaction_field_mapping()

    def __init_embeddings(self):
        if self.config.feature_repr_strategy == "emb":
            # embeddings
            if self.config.use_lset:
                if not self.config.use_shallow_embedding:
                    self.lset_embedding = nn.Sequential(
                        SinusoidalPosEmb1D(
                            self.hidden_dim * self.embed_dim_scale,
                            self.lvlset_embed_theta,
                        ),
                        nn.Linear(
                            self.hidden_dim * self.embed_dim_scale, self.hidden_dim
                        ),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )  # for each grid
                else:
                    self.lset_embedding = SinusoidalPosEmb1D(
                        self.hidden_dim, self.lvlset_embed_theta
                    )  # for each grid

            if not self.config.use_shallow_embedding:
                self.atom_charge_embedding = nn.Sequential(
                    SinusoidalPosEmb1D(
                        self.hidden_dim * self.embed_dim_scale, self.charge_embed_theta
                    ),
                    nn.Linear(self.hidden_dim * self.embed_dim_scale, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )  # for each atom
            else:
                self.atom_charge_embedding = SinusoidalPosEmb1D(
                    self.hidden_dim, self.charge_embed_theta
                )  # for each atom

    def __init_reaction_field_mapping(self):
        version = self.config.reaction_field_mapping_version
        if version == "pbgnn":
            self.__init_reaction_field_mapping_pbgnn()
        elif version == "pbgnn_fusion":
            self.__init_reaction_field_mapping_pbgnn_fusion()
        elif version == "pbgnn_bndy_fusion":
            self.__init_reaction_field_mapping_pbgnn_bndy_fusion()
        else:
            raise ValueError("Invalid reaction field mapping version")

    def __init_reaction_field_mapping_pbgnn(self):
        config_dict = asdict(self.config.pbgnn_config)

        # these properties are controlled by other parameters
        config_dict["input_dim"] = self.hidden_dim
        config = PBGNNConfig(**config_dict)
        self.pbgnn = PBGNN(config)

    def __init_reaction_field_mapping_pbgnn_fusion(self):
        config_dict = asdict(self.config.pbgnn_config)

        # these properties are controlled by other parameters
        config_dict["input_dim"] = self.hidden_dim
        config = PBGNNConfig(**config_dict)
        self.pbgnn = PBGNNwithCrossInteraction(config)

    def __init_reaction_field_mapping_pbgnn_bndy_fusion(self):
        config_dict = asdict(self.config.pbgnn_config)

        # these properties are controlled by other parameters
        config_dict["input_dim"] = self.hidden_dim
        config = PBGNNConfig(**config_dict)
        self.pbgnn = PBGNNwithBoundaryCrossInteraction(config)

    @property
    def in_channels(self):
        if self.config.feature_repr_strategy == "emb":
            return self.hidden_dim
        elif self.config.feature_repr_strategy == "concat":
            h = 1
            if self.config.use_lset:
                h += 1
            return h
        else:
            raise ValueError(
                f"Invalid feature representation strategy: {self.config.feature_repr_strategy}"
            )

    def atom_embedding(
        self,
        atom_charge: torch.Tensor,
        atom_type: torch.Tensor,
        atom_mask: torch.Tensor,
        grid_space: torch.Tensor,
    ):
        x = self.atom_charge_embedding(atom_charge * self.charge_embed_scale)
        return x

    def level_set_embedding(self, level_set: torch.Tensor):
        return self.lset_embedding(level_set * self.lvlset_embed_scale)

    def encode(
        self,
        voxel_level_set: torch.Tensor,
        atom_charge: torch.Tensor,
        atom_type: torch.Tensor,
        atom_mask: torch.Tensor,
        grid_space: torch.Tensor,
        batch_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        feat_encoding = {}
        if self.config.feature_repr_strategy == "emb":
            atom_emb = self.atom_embedding(
                atom_charge, atom_type, atom_mask, grid_space
            )
            # lset_emb = self.level_set_embedding(voxel_level_set)

            # compute reaction field potential
            # x = atom_emb + lset_emb

            feat_encoding["atom_emb"] = atom_emb
            if self.config.use_lset:
                lset_emb = self.level_set_embedding(voxel_level_set)
                feat_encoding["lset_emb"] = lset_emb

            if self.config.use_boundary_charge:
                feat_encoding["boundary_charge_emb"] = self.atom_embedding(
                    batch_dict["boundary_charge"], None, None, None
                )

            if self.config.use_boundary_lset:
                feat_encoding["boundary_lset_emb"] = self.level_set_embedding(
                    batch_dict["boundary_level_set"]
                )
        elif self.config.feature_repr_strategy == "concat":
            features = [
                atom_charge * self.charge_embed_scale,
            ]

            if self.config.use_lset:
                features.append(voxel_level_set * self.lvlset_embed_scale)

            x = torch.cat(
                features,
                dim=1,
            )
            feat_encoding["x"] = x
        else:
            raise ValueError(
                f"Invalid feature representation strategy: {self.config.feature_repr_strategy}"
            )
        return feat_encoding

    def reaction_field_mapping_pbgnn(
        self,
        encoding: Dict[str, torch.Tensor],
        atom_xyz: torch.Tensor,
        batch_of_atom_indices: torch.Tensor,
        batch_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        inputs = dict(
            atom_emb=encoding["atom_emb"],
            atom_xyz=atom_xyz,
            batch_of_atom_indices=batch_of_atom_indices,
            batch_dict=batch_dict,
            encoding=encoding,
        )
        dx = self.pbgnn(inputs)
        return dx

    def reaction_field_mapping(self, *args, **kwargs):
        version = self.config.reaction_field_mapping_version
        if version == "pbgnn":
            return self.reaction_field_mapping_pbgnn(*args, **kwargs)
        elif version == "pbgnn_fusion":
            return self.reaction_field_mapping_pbgnn(*args, **kwargs)
        elif version == "pbgnn_bndy_fusion":
            return self.reaction_field_mapping_pbgnn(*args, **kwargs)
        else:
            raise ValueError("Invalid reaction field mapping version")

    def compute_patch_epb(
        self, atom_charge, atom_mask, atom_potential, batch_of_atom_indices
    ):
        atom_bool_mask = atom_mask > 0
        unique_batch_idx = torch.unique(batch_of_atom_indices)
        smallest_batch_idx = unique_batch_idx.min()
        epb = torch.zeros(unique_batch_idx.shape[0], device=atom_charge.device)

        scattered_potential = atom_charge * atom_potential * atom_bool_mask.float()
        epb = (
            epb.scatter_add(
                0, batch_of_atom_indices - smallest_batch_idx, scattered_potential
            )
            * 0.5
        )
        return epb

    def forward(
        self,
        batch_of_atom_indices: torch.Tensor,
        level_set: torch.Tensor,
        atom_charge: torch.Tensor,
        atom_type: torch.Tensor,
        atom_mask: torch.Tensor,
        atom_xyz: torch.Tensor,
        grid_space: torch.Tensor,
        return_per_epb: bool = False,
        batch_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """for one voxel data

        :param voxel_x: P, C, D, H, W
        :return: P, D, H, W
        """
        encoding = self.encode(
            level_set,
            atom_charge,
            atom_type,
            atom_mask,
            grid_space,
            batch_dict=batch_dict,
        )
        per_epb = self.reaction_field_mapping(
            encoding, atom_xyz, batch_of_atom_indices, batch_dict=batch_dict
        )

        # predict epb
        if self.is_atom_wise_potential_trained:
            epb = self.compute_patch_epb(
                atom_charge, atom_mask, per_epb, batch_of_atom_indices
            )
        else:
            raise NotImplementedError

        outputs = (epb,)
        if return_per_epb:
            outputs += (per_epb,)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


if __name__ == "__main__":
    net = FullyConvMlsesModel(32, 96, 1)
    image = torch.randn(4, 96, 32, 32)
    output = net(image)
    print(output.shape)
