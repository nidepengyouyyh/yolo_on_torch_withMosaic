import numpy as np
import torch.nn as nn
import torch
from typing import Dict, Optional, Tuple, Union, Sequence, List
import math
from functools import partial
from torch import Tensor
from torch import distributed as torch_dist
import torch.nn.functional as F

from base.basebbox import BaseBoxes


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def build_padding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')

    # 根据填充类型实例化相应的层
    if padding_type == 'zero':
        return nn.ZeroPad2d(*args, **kwargs, **cfg_)
    elif padding_type == 'reflect':
        return nn.ReflectionPad2d(*args, **kwargs, **cfg_)
    elif padding_type == 'replicate':
        return nn.ReplicationPad2d(*args, **kwargs, **cfg_)
    else:
        raise KeyError(f'Unsupported padding type: {padding_type}')

def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    # 定义类型到 PyTorch nn 模块的映射
    conv_layers = {
        'Conv1d': nn.Conv1d,
        'Conv2d': nn.Conv2d,
        'Conv3d': nn.Conv3d,
        'Conv': nn.Conv2d,  # 默认使用 Conv2d
    }

    # 根据类型获取相应的卷积层
    if layer_type not in conv_layers:
        raise KeyError(f'Unsupported convolution type: {layer_type}')

    conv_layer = conv_layers[layer_type]

    # 实例化卷积层并返回
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer

def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    # 定义类型到 PyTorch nn 模块的映射
    norm_layers = {
        'BN': nn.BatchNorm2d,
        'SyncBN': nn.SyncBatchNorm,
        'GN': nn.GroupNorm,
        'IN': nn.InstanceNorm2d,
        'LN': nn.LayerNorm
    }

    # 根据类型获取相应的归一化层
    if layer_type not in norm_layers:
        raise KeyError(f'Unsupported normalization type: {layer_type}')

    norm_layer = norm_layers[layer_type]

    # 推断缩写形式
    abbr = layer_type.lower()

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    # 是否需要计算梯度
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)

    # 创建归一化层实例
    if layer_type == 'GN':
        if 'num_groups' not in cfg_:
            raise KeyError('The cfg dict must contain the key "num_groups" for GN')
        layer = norm_layer(num_channels=num_features, **cfg_)
    else:
        layer = norm_layer(num_features, **cfg_)

    # 设置参数的 requires_grad
    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
class SiLU(nn.Module):
    """Sigmoid Weighted Liner Unit."""

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, inputs) -> torch.Tensor:
        if self.inplace:
            return inputs.mul_(torch.sigmoid(inputs))
        else:
            return inputs * torch.sigmoid(inputs)

def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    layer_type = cfg.pop('type')

    # 激活层类型到 PyTorch 激活函数的映射
    activation_layers = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'PReLU': nn.PReLU,
        'RReLU': nn.RReLU,
        'ReLU6': nn.ReLU6,
        'ELU': nn.ELU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'SiLU': nn.SiLU if torch.__version__ >= '1.7.0' else SiLU,
    }

    # 根据类型获取对应的激活层类
    if layer_type not in activation_layers:
        raise KeyError(f'Unsupported activation type: {layer_type}')

    activation_layer = activation_layers[layer_type]

    # 实例化激活层
    layer = activation_layer(**cfg)

    return layer

def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor

def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def build_plugin_layer(cfg: Dict,
                       postfix: Union[int, str] = '',
                       **kwargs) -> Tuple[str, nn.Module]:
    """Build a plugin layer in PyTorch.

    Args:
        cfg (dict): cfg should contain:
            - type (str): identify plugin layer type.
            - layer args: args needed to instantiate a plugin layer.
        postfix (int, str): appended into abbreviation to create named layer.
            Default: ''.

    Returns:
        tuple[str, nn.Module]: The first one is the concatenation of
        abbreviation and postfix. The second is the created plugin layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    # Extract the type of the layer and its arguments
    layer_type = cfg.pop('type')
    layer_args = cfg.copy()

    # Use getattr to obtain the desired layer class from nn or your custom layers
    # It assumes that layer_type is a valid PyTorch layer or a custom one
    if hasattr(nn, layer_type):
        plugin_layer = getattr(nn, layer_type)
    else:
        raise KeyError(f'Layer type {layer_type} is not found in torch.nn')

    # Create abbreviation based on layer type
    abbr = layer_type[:3].lower()  # Use the first three letters as abbreviation

    assert isinstance(postfix, (int, str)), "Postfix should be an integer or string."
    name = abbr + str(postfix)

    # Instantiate the layer with the provided arguments and any additional kwargs
    layer = plugin_layer(**layer_args, **kwargs)

    return name, layer

# def gt_instances_preprocess(batch_gt_instances: Union[Tensor, Sequence],
#                             batch_size: int) -> Tensor:
#
#     assert isinstance(batch_gt_instances, Tensor)
#     box_dim = batch_gt_instances.size(-1) - 2
#     if len(batch_gt_instances) > 0:
#         gt_images_indexes = batch_gt_instances[:, 0]
#         max_gt_bbox_len = gt_images_indexes.unique(
#             return_counts=True)[1].max()
#         # fill zeros with length box_dim+1 if some shape of
#         # single batch not equal max_gt_bbox_len
#         batch_instance = torch.zeros(
#             (batch_size, max_gt_bbox_len, box_dim + 1),
#             dtype=batch_gt_instances.dtype,
#             device=batch_gt_instances.device)
#
#         for i in range(batch_size):
#             match_indexes = gt_images_indexes == i
#             gt_num = match_indexes.sum()
#             if gt_num:
#                 batch_instance[i, :gt_num] = batch_gt_instances[
#                     match_indexes, 1:]
#     else:
#         batch_instance = torch.zeros((batch_size, 0, box_dim + 1),
#                                      dtype=batch_gt_instances.dtype,
#                                      device=batch_gt_instances.device)
#
#     return batch_instance
def get_box_tensor(boxes: Union[Tensor, BaseBoxes]) -> Tensor:
    """Get tensor data from box type boxes.

    Args:
        boxes (Tensor or BaseBoxes): boxes with type of tensor or box type.
            If its type is a tensor, the boxes will be directly returned.
            If its type is a box type, the `boxes.tensor` will be returned.

    Returns:
        Tensor: boxes tensor.
    """
    if isinstance(boxes, BaseBoxes):
        boxes = boxes.tensor
    return boxes
def gt_instances_preprocess(batch_gt_instances: Union[Tensor, Sequence],
                            batch_size: int) -> Tensor:
    """Split batch_gt_instances with batch size.

    From [all_gt_bboxes, box_dim+2] to [batch_size, number_gt, box_dim+1].
    For horizontal box, box_dim=4, for rotated box, box_dim=5

    If some shape of single batch smaller than
    gt bbox len, then using zeros to fill.

    Args:
        batch_gt_instances (Sequence[Tensor]): Ground truth
            instances for whole batch, shape [all_gt_bboxes, box_dim+2]
        batch_size (int): Batch size.

    Returns:
        Tensor: batch gt instances data, shape
                [batch_size, number_gt, box_dim+1]
    """
    if isinstance(batch_gt_instances, Sequence):
        max_gt_bbox_len = max(
            [len(gt_instances) for gt_instances in batch_gt_instances])
        # fill zeros with length box_dim+1 if some shape of
        # single batch not equal max_gt_bbox_len
        batch_instance_list = []
        for index, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            labels = gt_instance.labels
            box_dim = get_box_tensor(bboxes).size(-1)
            batch_instance_list.append(
                torch.cat((labels[:, None], bboxes), dim=-1))

            if bboxes.shape[0] >= max_gt_bbox_len:
                continue

            fill_tensor = bboxes.new_full(
                [max_gt_bbox_len - bboxes.shape[0], box_dim + 1], 0)
            batch_instance_list[index] = torch.cat(
                (batch_instance_list[index], fill_tensor), dim=0)

        return torch.stack(batch_instance_list)
    else:
        # faster version
        # format of batch_gt_instances: [img_ind, cls_ind, (box)]
        # For example horizontal box should be:
        # [img_ind, cls_ind, x1, y1, x2, y2]
        # Rotated box should be
        # [img_ind, cls_ind, x, y, w, h, a]

        # sqlit batch gt instance [all_gt_bboxes, box_dim+2] ->
        # [batch_size, max_gt_bbox_len, box_dim+1]
        assert isinstance(batch_gt_instances, Tensor)
        box_dim = batch_gt_instances.size(-1) - 2
        if len(batch_gt_instances) > 0:
            gt_images_indexes = batch_gt_instances[:, 0]
            max_gt_bbox_len = gt_images_indexes.unique(
                return_counts=True)[1].max()
            # fill zeros with length box_dim+1 if some shape of
            # single batch not equal max_gt_bbox_len
            batch_instance = torch.zeros(
                (batch_size, max_gt_bbox_len, box_dim + 1),
                dtype=batch_gt_instances.dtype,
                device=batch_gt_instances.device)

            for i in range(batch_size):
                match_indexes = gt_images_indexes == i
                gt_num = match_indexes.sum()
                if gt_num:
                    batch_instance[i, :gt_num] = batch_gt_instances[
                        match_indexes, 1:]
        else:
            batch_instance = torch.zeros((batch_size, 0, box_dim + 1),
                                         dtype=batch_gt_instances.dtype,
                                         device=batch_gt_instances.device)

        return batch_instance

def get_dist_info(group = None) -> Tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size

def get_world_size(group = None) -> int:
    # if is_distributed():
    #     # handle low versions of torch like 1.5.0 which does not support
    #     # passing in None for group argument
    #     if group is None:
    #         group = get_default_group()
    #     return torch_dist.get_world_size(group)
    # else:
    #     return 1
    return 1

def get_rank(group = None) -> int:
    return 0

def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results

def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

def is_cuda_available() -> bool:
    """Returns True if cuda devices exist."""
    return torch.cuda.is_available()

def get_device() -> str:
    """Returns the currently existing device type.

    Returns:
        str: cuda | npu | mlu | mps | musa | cpu.
    """
    DEVICE = 'cpu'
    if is_cuda_available():
        DEVICE = 'cuda'
    return DEVICE

def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the models, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(
        tensor_list,
        list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({
        tensor.ndim
        for tensor in tensor_list
    }) == 1, (f'Expected the dimensions of all tensors must be the same, '
              f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)

try:
    from torchvision.ops import nms as torchvision_nms
    has_torchvision_nms = True
except ImportError:
    has_torchvision_nms = False


def nms(boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        offset: int = 0,
        score_threshold: float = 0,
        max_num: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is a GPU tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (np.ndarray): boxes in shape (N, 4).
        scores (np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
        score_threshold (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.

    Returns:
        tuple: kept dets (boxes and scores) and indices, which always have
        the same data type as the input.
    """
    assert isinstance(boxes, (Tensor, np.ndarray))
    assert isinstance(scores, (Tensor, np.ndarray))

    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)

    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    # 根据 torchvision 版本选择 NMS 方法
    if has_torchvision_nms:
        # 新版本 PyTorch 兼容方式
        inds = torchvision_nms(boxes, scores, iou_threshold)
    else:
        # 旧版本兼容处理方式
        # 检查是否使用旧的 torch.ops.torchvision.nms 调用方式
        if hasattr(torch.ops, 'torchvision') and hasattr(torch.ops.torchvision, 'nms'):
            inds = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
        else:
            raise ImportError("NMS function is not available in this version of torchvision or torch.")

    # 进行 score 阈值筛选
    if score_threshold > 0:
        inds = inds[scores[inds] > score_threshold]

    # 进行最大数量筛选
    if max_num > 0:
        inds = inds[:max_num]

    # 获取筛选后的检测框和分数
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)

    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()

    return dets, inds

def add_pred_to_datasample(data_samples,
                           results_list):
    for data_sample, pred_instances in zip(data_samples, results_list):
        data_sample.pred_instances = pred_instances
    samplelist_boxtype2tensor(data_samples)
    return data_samples

def samplelist_boxtype2tensor(batch_data_samples):
    for data_samples in batch_data_samples:
        if 'gt_instances' in data_samples:
            bboxes = data_samples.gt_instances.get('bboxes', None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.gt_instances.bboxes = bboxes.tensor
        if 'pred_instances' in data_samples:
            bboxes = data_samples.pred_instances.get('bboxes', None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.pred_instances.bboxes = bboxes.tensor
        if 'ignored_instances' in data_samples:
            bboxes = data_samples.ignored_instances.get('bboxes', None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.ignored_instances.bboxes = bboxes.tensor

def load_weights_with_mapping(model, weight_path):
    # 加载权重文件
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # # 创建一个新的字典用于存储映射后的权重
    # new_state_dict = {}
    # for model_key in models.state_dict().keys():
    #     # 获取权重文件中对应的键名
    #     # checkpoint_key = model_key.replace('stem.', 'backbone.stem.')
    #
    #     if model_key in model_weights:
    #         new_state_dict[model_key] = model_weights[model_key]
    #     else:
    #         print(f"{model_key}: Not found in weight file.")

    # 加载映射后的权重
    model.load_state_dict(model_weights, strict=False)
    return model