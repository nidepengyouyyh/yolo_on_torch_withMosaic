import copy
import logging
from collections import defaultdict
from typing import Union, List, Tuple
from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch import Tensor



class BaseModule(nn.Module):
    """Base module for all modules with parameter initialization functionality.

    This is a simplified version of the original `BaseModule` for general PyTorch usage.

    Args:
        init_cfg (dict or List[dict], optional): Initialization config dict.
    """

    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`."""
        super().__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    @is_init.setter
    def is_init(self, value):
        self._is_init = value

    def init_weights(self):
        """Initialize the weights."""
        is_top_level_module = not hasattr(self, '_params_init_info')

        # If this is the top-level module, initialize `_params_init_info`
        if is_top_level_module:
            self._params_init_info = defaultdict(dict)
            for name, param in self.named_parameters():
                self._params_init_info[param]['init_info'] = (
                    f'The value is the same before and after calling `init_weights` '
                    f'of {self.__class__.__name__}')
                self._params_init_info[param]['tmp_mean_value'] = param.data.mean().item()

            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                logging.info(f'Initializing {module_name} with init_cfg {self.init_cfg}')

                # Support for list or dict configurations
                init_cfgs = self.init_cfg if isinstance(self.init_cfg, list) else [self.init_cfg]
                other_cfgs = [cfg for cfg in init_cfgs if cfg.get('type') != 'Pretrained']
                pretrained_cfgs = [cfg for cfg in init_cfgs if cfg.get('type') == 'Pretrained']

                # Initialize other weights first
                self.apply_initialization(other_cfgs)

                # Initialize submodules
                for m in self.children():
                    if hasattr(m, 'init_weights') and not m.is_init:
                        m.init_weights()

                # Apply pretrained weights if available
                if pretrained_cfgs:
                    self.apply_initialization(pretrained_cfgs)

                self._is_init = True
            else:
                logging.warning(f'No `init_cfg` provided for {module_name}, using default initialization.')
        else:
            logging.warning(f'init_weights of {module_name} has been called more than once.')

        for m in self.children():
            if hasattr(m, 'init_weights') and not getattr(
                    m, 'is_init', False):
                m.init_weights()


        if is_top_level_module:
            self._log_init_info()

            for sub_module in self.modules():
                del sub_module._params_init_info

    def apply_initialization(self, init_cfgs):
        """Apply initialization based on provided configurations."""
        for init_cfg in init_cfgs:
            # Here, add the logic to apply initialization methods (e.g., 'kaiming', 'xavier', etc.)
            init_type = init_cfg.get('type', 'kaiming')
            if init_type == 'kaiming':
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == 'xavier':
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_normal_(m.weight)
            # Add more initialization methods if needed

    def _log_init_info(self):
        """Log initialization information."""
        for name, param in self.named_parameters():
            logging.info(f'{name} - Shape: {param.shape}, '
                         f'Init Info: {self._params_init_info[param]["init_info"]}')

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s


# class BaseDenseHead(BaseModule, metaclass=ABCMeta):
#     def __init__(self, init_cfg=None):
#         super().__init__(init_cfg=init_cfg)
#         self._raw_positive_infos = dict()
#
#     def init_weights(self):
#         super().init_weights()
#         for m in self.modules():
#             if hasattr(m, 'conv_offset')
#                 nn.init.constant_(m.conv_offset, 0)
#     def get_positive_infos(self):
#         if len(self._raw_positive_infos) == 0:
#             return None
#
#         sampling_results = self._raw_positive_infos.get('sampling_results', None)
#         assert sampling_results is not None
#         positive_infos = []
#         for sampling_result in sampling_results:
#             pos_info = InstanceData()
#             pos_info.bboxes = sampling_result.pos_gt_bboxes
#             pos_info.labels = sampling_result.pos_gt_labels
#             pos_info.priors = sampling_result.pos_priors
#             pos_info.pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
#             pos_info.pos_inds = sampling_result.pos_inds
#             positive_infos.append(pos_info)
#         return positive_infos
#
#     def loss(self, x: Tuple[Tensor], batch_data_samples):
#         outs = self(x)
#
#         outputs = unpack_gt_instances(batch_data_samples)
#         (batch_gt_instances, batch_gt_instances_ignore,
#          batch_img_metas) = outputs
#
#         loss_inputs = outs + (batch_gt_instances, batch_img_metas,
#                               batch_gt_instances_ignore)
#         losses = self.loss_by_feat(*loss_inputs)
#         return losses
#
#     @abstractmethod
#     def loss_by_feat(self, **kwargs) -> dict:
#         """Calculate the loss based on the features extracted by the detection
#         head."""
#         pass
#
#     def loss_and_predict(self, x, batch_data_samples, **kwargs):
#         """Calculate losses and make predictions from a batch of inputs and data samples."""
#         batch_gt_instances = [sample.gt_instances for sample in batch_data_samples]
#         batch_img_metas = [sample.metainfo for sample in batch_data_samples]
#
#         outs = self.forward(x)
#
#         loss_inputs = outs + (batch_gt_instances, batch_img_metas)
#         losses = self.loss_by_feat(*loss_inputs, **kwargs)
#
#         predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, **kwargs)
#         return losses, predictions