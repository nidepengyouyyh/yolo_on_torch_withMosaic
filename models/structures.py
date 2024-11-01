import copy
import itertools
from typing import Optional, Iterator, Tuple, Any, Union, Type, Sized, List, Sequence, Dict
import numpy as np
import torch
from base.basebbox import BaseBoxes
from abc import ABCMeta, abstractmethod

BoolTypeTensor: Union[Any]
BoolTypeTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]

class BaseDataElement:

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """Set or change key-value pairs in ``metainfo_field`` by parameter
        ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        assert isinstance(
            metainfo,
            dict), f'metainfo should be a ``dict`` but got {type(metainfo)}'
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type='metainfo', dtype=None)

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter
        ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                models predictions.
        """
        assert isinstance(data,
                          dict), f'data should be a `dict` but got {data}'
        for k, v in data.items():
            # Use `setattr()` rather than `self.set_field` to allow `set_data`
            # to set property method.
            setattr(self, k, v)

    def update(self, instance: 'BaseDataElement') -> None:
        """The update() method updates the BaseDataElement with the elements
        from another BaseDataElement object.

        Args:
            instance (BaseDataElement): Another BaseDataElement object for
                update the current object.
        """
        assert isinstance(
            instance, BaseDataElement
        ), f'instance should be a `BaseDataElement` but got {type(instance)}'
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self,
            *,
            metainfo: Optional[dict] = None,
            **kwargs) -> 'BaseDataElement':
        """Return a new data element with same type. If ``metainfo`` and
        ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new result will overwrite it
        with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
                Defaults to None.
            kwargs (dict): A dict contains annotations of image or
                models predictions.

        Returns:
            BaseDataElement: A new data element with same type.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        """Deep copy the current data element.

        Returns:
            BaseDataElement: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        # We assume that the name of the attribute related to property is
        # '_' + the name of the property. We use this rule to filter out
        # private keys.
        # TODO: Use a more robust way to solve this problem
        private_keys = {
            '_' + key
            for key in self._data_fields
            if isinstance(getattr(type(self), key, None), property)
        }
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return list(self._metainfo_fields)

    def values(self) -> list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        """dict: A dict contains metainfo of current data element."""
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):
        """setattr is only used to set data."""
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')
        else:
            self.set_field(
                name=name, value=value, field_type='data', dtype=None)

    def __delattr__(self, item: str):
        """Delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a '
                                 'private attribute, which is immutable.')
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __delitem__ = __delattr__

    def get(self, key, default=None) -> Any:
        """Get property in data and metainfo as the same as python."""
        # Use `getattr()` rather than `self.__dict__.get()` to allow getting
        # properties.
        return getattr(self, key, default)

    def pop(self, *args) -> Any:
        """Pop property in data and metainfo as the same as python."""
        assert len(args) < 3, '``pop`` get more than 2 arguments'
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f'{args[0]} is not contained in metainfo or data')

    def __contains__(self, item: str) -> bool:
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(self,
                  value: Any,
                  name: str,
                  dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
                  field_type: str = 'data') -> None:
        """Special method for set union field, used as property.setter
        functions."""
        assert field_type in ['metainfo', 'data']
        if dtype is not None:
            assert isinstance(
                value,
                dtype), f'{value} should be a {dtype} but got {type(value)}'

        if field_type == 'metainfo':
            if name in self._data_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of metainfo '
                    f'because {name} is already a data field')
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of data '
                    f'because {name} is already a metainfo field')
            self._data_fields.add(name)
        super().__setattr__(name, value)

    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cpu(self) -> 'BaseDataElement':
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cuda(self) -> 'BaseDataElement':
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def musa(self) -> 'BaseDataElement':
        """Convert all tensors to musa in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.musa()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def npu(self) -> 'BaseDataElement':
        """Convert all tensors to NPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.npu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def mlu(self) -> 'BaseDataElement':
        """Convert all tensors to MLU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.mlu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def detach(self) -> 'BaseDataElement':
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def numpy(self) -> 'BaseDataElement':
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> 'BaseDataElement':
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        """Convert BaseDataElement to dict."""
        return {
            k: v.to_dict() if isinstance(v, BaseDataElement) else v
            for k, v in self.all_items()
        }

    def __repr__(self) -> str:
        """Represent the object."""

        def _addindent(s_: str, num_spaces: int) -> str:
            """This func is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)  # type: ignore
            s = first + '\n' + s  # type: ignore
            return s  # type: ignore

        def dump(obj: Any) -> str:
            """Represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str.
            """
            _repr = ''
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f'\n{k}: {_addindent(dump(v), 4)}'
            elif isinstance(obj, BaseDataElement):
                _repr += '\n\n    META INFORMATION'
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += '\n\n    DATA FIELDS'
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f'<{classname}({_repr}\n) at {hex(id(obj))}>'
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)

class InstanceData(BaseDataElement):

    def __setattr__(self, name: str, value: Sized):
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `InstanceData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value,
                              Sized), 'value must contain `__len__` attribute'

            if len(self) > 0:
                assert len(value) == len(self), 'The length of ' \
                                                f'values {len(value)} is ' \
                                                'not consistent with ' \
                                                'the length of this ' \
                                                ':obj:`InstanceData` ' \
                                                f'{len(self)}'
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item) -> 'InstanceData':
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`InstanceData`: Corresponding values.
        """
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # More details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, 'Only support to get the' \
                                    ' values along the first dimension.'
            if isinstance(item, BoolTypeTensor.__args__):
                assert len(item) == len(self), 'The shape of the ' \
                                               'input(BoolTensor) ' \
                                               f'{len(item)} ' \
                                               'does not match the shape ' \
                                               'of the indexed tensor ' \
                                               'in results_field ' \
                                               f'{len(self)} at ' \
                                               'first dimension.'

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(
                        v, (str, list, tuple)) or (hasattr(v, '__getitem__')
                                                   and hasattr(v, 'cat')):
                    # convert to indexes from BoolTensor
                    if isinstance(item, BoolTypeTensor.__args__):
                        indexes = torch.nonzero(item).view(
                            -1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f'The type of `{k}` is `{type(v)}`, which has no '
                        'attribute of `cat`, so it does not '
                        'support slice with `bool`')

        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type:ignore

    @staticmethod
    def cat(instances_list: List['InstanceData']) -> 'InstanceData':
        """Concat the instances of all :obj:`InstanceData` in the list.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`InstanceData`]): A list
                of :obj:`InstanceData`.

        Returns:
            :obj:`InstanceData`
        """
        assert all(
            isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [
            instances.all_keys() for instances in instances_list
        ]
        assert len({len(field_keys) for field_keys in field_keys_list}) \
               == 1 and len(set(itertools.chain(*field_keys_list))) \
               == len(field_keys_list[0]), 'There are different keys in ' \
                                           '`instances_list`, which may ' \
                                           'cause the cat operation ' \
                                           'to fail. Please make sure all ' \
                                           'elements in `instances_list` ' \
                                           'have the exact same key.'

        new_data = instances_list[0].__class__(
            metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                new_values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            elif hasattr(v0, 'cat'):
                new_values = v0.cat(values)
            else:
                raise ValueError(
                    f'The type of `{k}` is `{type(v0)}` which has no '
                    'attribute of `cat`')
            new_data[k] = new_values
        return new_data  # type:ignore

    def __len__(self) -> int:
        """int: The length of InstanceData."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0


class PixelData(BaseDataElement):
    """Data structure for pixel-level annotations or predictions.

    All data items in ``data_fields`` of ``PixelData`` meet the following
    requirements:

    - They all have 3 dimensions in orders of channel, height, and width.
    - They should have the same height and width.

    Examples:
        >>> metainfo = dict(
        ...     img_id=random.randint(0, 100),
        ...     img_shape=(random.randint(400, 600), random.randint(400, 600)))
        >>> image = np.random.randint(0, 255, (4, 20, 40))
        >>> featmap = torch.randint(0, 255, (10, 20, 40))
        >>> pixel_data = PixelData(metainfo=metainfo,
        ...                        image=image,
        ...                        featmap=featmap)
        >>> print(pixel_data.shape)
        (20, 40)

        >>> # slice
        >>> slice_data = pixel_data[10:20, 20:40]
        >>> assert slice_data.shape == (10, 20)
        >>> slice_data = pixel_data[10, 20]
        >>> assert slice_data.shape == (1, 1)

        >>> # set
        >>> pixel_data.map3 = torch.randint(0, 255, (20, 40))
        >>> assert tuple(pixel_data.map3.shape) == (1, 20, 40)
        >>> with self.assertRaises(AssertionError):
        ...     # The dimension must be 3 or 2
        ...     pixel_data.map2 = torch.randint(0, 255, (1, 3, 20, 40))
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """Set attributes of ``PixelData``.

        If the dimension of value is 2 and its shape meet the demand, it
        will automatically expand its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `PixelData`.
            value (Union[torch.Tensor, np.ndarray]): The value to store in.
                The type of value must be `torch.Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `PixelData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), \
                f'Can not set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    'The height and width of '
                    f'values {tuple(value.shape[-2:])} is '
                    'not consistent with '
                    'the shape of this '
                    ':obj:`PixelData` '
                    f'{self.shape}')
            assert value.ndim in [
                2, 3
            ], f'The dim of value must be 2 or 3, but got {value.ndim}'
            if value.ndim == 2:
                value = value[None]
            super().__setattr__(name, value)

    # TODO torch.Long/bool
    def __getitem__(self, item: Sequence[Union[int, slice]]) -> 'PixelData':
        """
        Args:
            item (Sequence[Union[int, slice]]): Get the corresponding values
                according to item.

        Returns:
            :obj:`PixelData`: Corresponding values.
        """

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):

            assert len(item) == 2, 'Only support to slice height and width'
            tmp_item: List[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(
                        0, slice(single_item, None, self.shape[-index - 1]))
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError(
                        'The type of element in input must be int or slice, '
                        f'but got {type(single_item)}')
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError(
                f'Unsupported type {type(item)} for slicing PixelData')
        return new_data

    @property
    def shape(self):
        """The shape of pixel data."""
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape[-2:])
        else:
            return None

    # TODO padding, resize


box_types: dict = {}
_box_type_to_name: dict = {}
box_converters: dict = {}
def get_box_type(box_type: Union[str, type]) -> Tuple[str, type]:
    """get both box type name and class.

    Args:
        box_type (str or type): Single box type name or class.

    Returns:
        Tuple[str, type]: A tuple of box type name and class.
    """
    if isinstance(box_type, str):
        type_name = box_type.lower()
        assert type_name in box_types, \
            f"Box type {type_name} hasn't been registered in box_types."
        type_cls = box_types[type_name]
    elif issubclass(box_type, BaseBoxes):
        assert box_type in _box_type_to_name, \
            f"Box type {box_type} hasn't been registered in box_types."
        type_name = _box_type_to_name[box_type]
        type_cls = box_type
    else:
        raise KeyError('box_type must be a str or class inheriting from '
                       f'BaseBoxes, but got {type(box_type)}.')
    return type_name, type_cls



class BaseOptimWrapper(metaclass=ABCMeta):

    def __init__(self, optimizer):
        self.optimizer = optimizer

        # The Following code is used to initialize `base_param_settings`.
        # `base_param_settings` is used to store the parameters that are not
        # updated by the optimizer.
        # The `base_param_settings` used for tracking the base learning in the
        # optimizer. If the optimizer has multiple parameter groups, this
        # params will not be scaled by the loss factor.
        if len(optimizer.param_groups) > 1:
            self.base_param_settings = {
                'params': torch.tensor([0.0], dtype=torch.float)
            }
            self.base_param_settings.update(**self.optimizer.defaults)
        else:
            self.base_param_settings = None  # type: ignore

    @abstractmethod
    def update_params(self, *args, **kwargs):
        """Update parameters in :attr:`optimizer`."""

    @abstractmethod
    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation."""

    @abstractmethod
    def zero_grad(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.zero_grad``."""

    @abstractmethod
    def step(self, **kwargs):
        """Call the step method of optimizer."""

    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``."""
        state_dict = self.optimizer.state_dict()
        if self.base_param_settings is not None:
            state_dict['base_param_settings'] = self.base_param_settings
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """A wrapper of ``Optimizer.load_state_dict``. load the state dict of
        :attr:`optimizer`.

        Provide unified ``load_state_dict`` interface compatible with automatic
        mixed precision training. Subclass can overload this method to
        implement the required logic. For example, the state dictionary of
        GradScaler should be loaded when training with ``torch.cuda.amp``.

        Args:
            state_dict (dict): The state dictionary of :attr:`optimizer`.
        """
        base_param_settings = state_dict.pop('base_param_settings', None)

        if base_param_settings is not None:
            self.base_param_settings = base_param_settings

        # load state_dict of optimizer
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self) -> List[dict]:
        """A wrapper of ``Optimizer.param_groups``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        if self.base_param_settings is not None:
            return self.optimizer.param_groups + [self.base_param_settings]
        else:
            return self.optimizer.param_groups

    @property
    def defaults(self) -> dict:
        """A wrapper of ``Optimizer.defaults``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.defaults

    def get_lr(self):
        """Get the learning rate of the optimizer.

        Provide unified interface to get learning rate of optimizer.

        Returns:
            Dict[str, List[float]]:
            param_groups learning rate of the optimizer.
        """
        res = {}
        if self.base_param_settings is not None:
            res['base_lr'] = [self.base_param_settings['lr']]

        res['lr'] = [group['lr'] for group in self.optimizer.param_groups]

        return res

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of the optimizer.

        Provide unified interface to get momentum of optimizer.

        Returns:
            Dict[str, List[float]]: Momentum of the optimizer.
        """
        momentum = []
        for group in self.optimizer.param_groups:
            # Get momentum of SGD.
            if 'momentum' in group.keys():
                momentum.append(group['momentum'])
            # Get momentum of Adam.
            elif 'betas' in group.keys():
                momentum.append(group['betas'][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)