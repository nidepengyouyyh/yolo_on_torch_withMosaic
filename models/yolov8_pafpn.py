import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from typing import List, Union
from .basemodule import BaseModule
from models.networks import ConvModule, CSPLayerWithTwoConv
from utils import make_divisible, make_round


class BaseYOLONeck(BaseModule, metaclass=ABCMeta):
    """Base neck used in YOLO series, implemented in pure PyTorch."""

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[int, List[int]],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 upsample_feats_cat_first: bool = True,
                 freeze_all: bool = False,
                 norm_cfg: dict = None,
                 act_cfg: dict = None,
                 init_cfg: dict = None,):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = upsample_feats_cat_first
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # Build reduce layers
        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(self.build_reduce_layer(idx))

        # Build top-down blocks
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        # Build bottom-up blocks
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        # Build output layers
        self.out_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.out_layers.append(self.build_out_layer(idx))

        if self.freeze_all:
            self._freeze_all()

    @abstractmethod
    def build_reduce_layer(self, idx: int):
        """Build reduce layer."""
        pass

    @abstractmethod
    def build_upsample_layer(self, idx: int):
        """Build upsample layer."""
        pass

    @abstractmethod
    def build_top_down_layer(self, idx: int):
        """Build top down layer."""
        pass

    @abstractmethod
    def build_downsample_layer(self, idx: int):
        """Build downsample layer."""
        pass

    @abstractmethod
    def build_bottom_up_layer(self, idx: int):
        """Build bottom up layer."""
        pass

    @abstractmethod
    def build_out_layer(self, idx: int):
        """Build output layer."""
        pass

    def _freeze_all(self):
        """Freeze all layers in the models."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Set the models to training mode, but keep the normalization layers frozen if required."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # Reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # Top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)

            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], dim=1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], dim=1)

            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # Bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        # Output layers
        results = [self.out_layers[idx](outs[idx]) for idx in range(len(self.in_channels))]

        return tuple(results)

class YOLOv8PAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOv5.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all (bool): Whether to freeze the models.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: dict = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: dict = None):
        self.num_csp_blocks = num_csp_blocks
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg
        )

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.reset_parameters()
        else:
            super().init_weights()

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """Build reduce layer.

        Args:
            idx (int): Layer index.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """Build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_top_down_layer(self, idx: int):
        """Build top-down layer.

        Args:
            idx (int): Layer index.

        Returns:
            nn.Module: The top-down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """Build downsample layer.

        Args:
            idx (int): Layer index.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """Build bottom-up layer.

        Args:
            idx (int): Layer index.

        Returns:
            nn.Module: The bottom-up layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """Build output layer."""
        return nn.Identity()