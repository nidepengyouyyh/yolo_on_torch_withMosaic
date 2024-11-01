from typing import Optional

from torch import nn

from models.optim_wrapper import OptimWrapper
from torch.distributed import ProcessGroup
from torch import distributed as torch_dist
from torch.optim import SGD, AdamW
def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()
def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()
def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1
class YOLOv5OptimizerConstructor:
    """YOLOv5 constructor for optimizers.

    It has the following functions：

        - divides the optimizer parameters into 3 groups:
        Conv, Bias and BN

        - support `weight_decay` parameter adaption based on
        `batch_size_per_gpu`

    Args:
        optim_wrapper_cfg (dict): The config dict of the optimizer wrapper.
            Positional fields are

                - ``type``: class name of the OptimizerWrapper
                - ``optimizer``: The configuration of optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer wrapper type,
                  e.g., accumulative_counts, clip_grad, etc.

            The positional fields of ``optimizer`` are

                - `type`: class name of the optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.

        paramwise_cfg (dict, optional): Parameter-wise options. Must include
            `base_total_batch_size` if not None. If the total input batch
            is smaller than `base_total_batch_size`, the `weight_decay`
            parameter will be kept unchanged, otherwise linear scaling.

    Example:
        >>> models = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optim_wrapper_cfg = dict(
        >>>     dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01,
        >>>         momentum=0.9, weight_decay=0.0001, batch_size_per_gpu=16))
        >>> paramwise_cfg = dict(base_total_batch_size=64)
        >>> optim_wrapper_builder = YOLOv5OptimizerConstructor(
        >>>     optim_wrapper_cfg, paramwise_cfg)
        >>> optim_wrapper = optim_wrapper_builder(models)
    """

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        if paramwise_cfg is None:
            paramwise_cfg = {'base_total_batch_size': 64}
        assert 'base_total_batch_size' in paramwise_cfg

        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optim_wrapper_cfg)}')
        # assert 'optimizer' in optim_wrapper_cfg, (
        #     '`optim_wrapper_cfg` must contain "optimizer" config')

        self.optim_wrapper_cfg = optim_wrapper_cfg
        self.optimizer_cfg = self.optim_wrapper_cfg.pop('optimizer')
        self.base_total_batch_size = paramwise_cfg['base_total_batch_size']

    def __call__(self, model: nn.Module) -> OptimWrapper:
        optimizer_cfg = self.optimizer_cfg.copy()
        weight_decay = optimizer_cfg.pop('weight_decay', 0)

        if 'batch_size_per_gpu' in optimizer_cfg:
            batch_size_per_gpu = optimizer_cfg.pop('batch_size_per_gpu')
            # No scaling if total_batch_size is less than
            # base_total_batch_size, otherwise linear scaling.
            total_batch_size = get_world_size() * batch_size_per_gpu
            accumulate = max(
                round(self.base_total_batch_size / total_batch_size), 1)
            scale_factor = total_batch_size * \
                accumulate / self.base_total_batch_size

            if scale_factor != 1:
                weight_decay *= scale_factor

        params_groups = [], [], []

        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                params_groups[2].append(v.bias)
            # Includes SyncBatchNorm
            if isinstance(v, nn.modules.batchnorm._NormBase):
                params_groups[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                params_groups[0].append(v.weight)

        # Note: Make sure bias is in the last parameter group
        optimizer_cfg['params'] = []
        # conv
        optimizer_cfg['params'].append({
            'params': params_groups[0],
            'weight_decay': weight_decay
        })
        # bn
        optimizer_cfg['params'].append({'params': params_groups[1]})
        # bias
        optimizer_cfg['params'].append({'params': params_groups[2]})

        del params_groups

        optimizer_type = optimizer_cfg.pop('type').lower()
        # print(optimizer_type)
        if optimizer_type == 'sgd':
            optimizer = SGD(params=optimizer_cfg['params'], lr=optimizer_cfg['lr'],
                            momentum=optimizer_cfg.get('momentum', 0.9),
                            weight_decay=weight_decay,
                            nesterov=optimizer_cfg.get('nesterov', False))
        elif optimizer_type == 'adamw':
            optimizer = AdamW(params=optimizer_cfg['params'], lr=optimizer_cfg['lr'],
                              betas=optimizer_cfg.get('betas', (0.9, 0.999)),
                              eps=optimizer_cfg.get('eps', 1e-8),
                              weight_decay=weight_decay,
                              amsgrad=optimizer_cfg.get('amsgrad', False))
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        optim_wrapper = OptimWrapper(optimizer=optimizer, clip_grad=self.optim_wrapper_cfg['clip_grad'])
        return optim_wrapper