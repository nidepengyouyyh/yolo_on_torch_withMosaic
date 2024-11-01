import functools
from collections import defaultdict, abc as container_abcs
from contextlib import contextmanager
from copy import deepcopy, copy
from itertools import chain
from typing import Optional, Type, TypeVar, Sequence, Union, Tuple, Callable, Any, Dict, List
from collections import OrderedDict
import inspect

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import required
import threading
from models.structures import BaseOptimWrapper
TORCH_VERSION = torch.__version__

def _get_norm() -> tuple:
    """A wrapper to obtain base classes of normalization layers from PyTorch or
    Parrots."""
    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.instancenorm import _InstanceNorm
    SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_
def has_batch_norm(model: nn.Module) -> bool:
    """Detect whether models has a BatchNormalization layer.

    Args:
        model (nn.Module): training models.

    Returns:
        bool: whether models has a BatchNormalization layer
    """
    if isinstance(model, _BatchNorm):
        return True
    for m in model.children():
        if has_batch_norm(m):
            return True
    return False
_BatchNorm = _get_norm()
class HistoryBuffer:
    """Unified storage format for different log types.

    ``HistoryBuffer`` records the history of log for further statistics.

    Examples:
        >>> history_buffer = HistoryBuffer()
        >>> # Update history_buffer.
        >>> history_buffer.update(1)
        >>> history_buffer.update(2)
        >>> history_buffer.min()  # minimum of (1, 2)
        1
        >>> history_buffer.max()  # maximum of (1, 2)
        2
        >>> history_buffer.mean()  # mean of (1, 2)
        1.5
        >>> history_buffer.statistics('mean')  # access method by string.
        1.5

    Args:
        log_history (Sequence): History logs. Defaults to [].
        count_history (Sequence): Counts of history logs. Defaults to [].
        max_length (int): The max length of history logs. Defaults to 1000000.
    """
    _statistics_methods: dict = dict()

    def __init__(self,
                 log_history: Sequence = [],
                 count_history: Sequence = [],
                 max_length: int = 1000000):

        self.max_length = max_length
        self._set_default_statistics()
        assert len(log_history) == len(count_history), \
            'The lengths of log_history and count_histroy should be equal'
        if len(log_history) > max_length:
            self._log_history = np.array(log_history[-max_length:])
            self._count_history = np.array(count_history[-max_length:])
        else:
            self._log_history = np.array(log_history)
            self._count_history = np.array(count_history)

    def _set_default_statistics(self) -> None:
        """Register default statistic methods: min, max, current and mean."""
        self._statistics_methods.setdefault('min', HistoryBuffer.min)
        self._statistics_methods.setdefault('max', HistoryBuffer.max)
        self._statistics_methods.setdefault('current', HistoryBuffer.current)
        self._statistics_methods.setdefault('mean', HistoryBuffer.mean)

    def update(self, log_val: Union[int, float], count: int = 1) -> None:
        """update the log history.

        If the length of the buffer exceeds ``self._max_length``, the oldest
        element will be removed from the buffer.

        Args:
            log_val (int or float): The value of log.
            count (int): The accumulation times of log, defaults to 1.
            ``count`` will be used in smooth statistics.
        """
        if (not isinstance(log_val, (int, float))
                or not isinstance(count, (int, float))):
            raise TypeError(f'log_val must be int or float but got '
                            f'{type(log_val)}, count must be int but got '
                            f'{type(count)}')
        self._log_history = np.append(self._log_history, log_val)
        self._count_history = np.append(self._count_history, count)
        if len(self._log_history) > self.max_length:
            self._log_history = self._log_history[-self.max_length:]
            self._count_history = self._count_history[-self.max_length:]

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the ``_log_history`` and ``_count_history``.

        Returns:
            Tuple[np.ndarray, np.ndarray]: History logs and the counts of
            the history logs.
        """
        return self._log_history, self._count_history

    @classmethod
    def register_statistics(cls, method: Callable) -> Callable:
        """Register custom statistics method to ``_statistics_methods``.

        The registered method can be called by ``history_buffer.statistics``
        with corresponding method name and arguments.

        Examples:
            >>> @HistoryBuffer.register_statistics
            >>> def weighted_mean(self, window_size, weight):
            >>>     assert len(weight) == window_size
            >>>     return (self._log_history[-window_size:] *
            >>>             np.array(weight)).sum() / \
            >>>             self._count_history[-window_size:]

            >>> log_buffer = HistoryBuffer([1, 2], [1, 1])
            >>> log_buffer.statistics('weighted_mean', 2, [2, 1])
            2

        Args:
            method (Callable): Custom statistics method.
        Returns:
            Callable: Original custom statistics method.
        """
        method_name = method.__name__
        assert method_name not in cls._statistics_methods, \
            'method_name cannot be registered twice!'
        cls._statistics_methods[method_name] = method
        return method

    def statistics(self, method_name: str, *arg, **kwargs) -> Any:
        """Access statistics method by name.

        Args:
            method_name (str): Name of method.

        Returns:
            Any: Depends on corresponding method.
        """
        if method_name not in self._statistics_methods:
            raise KeyError(f'{method_name} has not been registered in '
                           'HistoryBuffer._statistics_methods')
        method = self._statistics_methods[method_name]
        # Provide self arguments for registered functions.
        return method(self, *arg, **kwargs)

    def mean(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the mean of the latest ``window_size`` values in log
        histories.

        If ``window_size is None`` or ``window_size > len(self._log_history)``,
        return the global mean value of history logs.

        Args:
            window_size (int, optional): Size of statistics window.
        Returns:
            np.ndarray: Mean value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), \
                'The type of window size should be int, but got ' \
                f'{type(window_size)}'
        else:
            window_size = len(self._log_history)
        logs_sum = self._log_history[-window_size:].sum()
        counts_sum = self._count_history[-window_size:].sum()
        return logs_sum / counts_sum

    def max(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the maximum value of the latest ``window_size`` values in log
        histories.

        If ``window_size is None`` or ``window_size > len(self._log_history)``,
        return the global maximum value of history logs.

        Args:
            window_size (int, optional): Size of statistics window.
        Returns:
            np.ndarray: The maximum value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), \
                'The type of window size should be int, but got ' \
                f'{type(window_size)}'
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].max()

    def min(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the minimum value of the latest ``window_size`` values in log
        histories.

        If ``window_size is None`` or ``window_size > len(self._log_history)``,
        return the global minimum value of history logs.

        Args:
            window_size (int, optional): Size of statistics window.
        Returns:
            np.ndarray: The minimum value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), \
                'The type of window size should be int, but got ' \
                f'{type(window_size)}'
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].min()

    def current(self) -> np.ndarray:
        """Return the recently updated values in log histories.

        Returns:
            np.ndarray: Recently updated values in log histories.
        """
        if len(self._log_history) == 0:
            raise ValueError('HistoryBuffer._log_history is an empty array! '
                             'please call update first')
        return self._log_history[-1]

    def __getstate__(self) -> dict:
        """Make ``_statistics_methods`` can be resumed.

        Returns:
            dict: State dict including statistics_methods.
        """
        self.__dict__.update(statistics_methods=self._statistics_methods)
        return self.__dict__

    def __setstate__(self, state):
        """Try to load ``_statistics_methods`` from state.

        Args:
            state (dict): State dict.
        """
        statistics_methods = state.pop('statistics_methods', {})
        self._set_default_statistics()
        self._statistics_methods.update(statistics_methods)
        self.__dict__.update(state)
T = TypeVar('T')
_lock = threading.RLock()
def _accquire_lock() -> None:
    """Acquire the module-level lock for serializing access to shared data.

    This should be released with _release_lock().
    """
    if _lock:
        _lock.acquire()

def _release_lock() -> None:
    """Release the module-level lock acquired by calling _accquire_lock()."""
    if _lock:
        _lock.release()
class ManagerMeta(type):
    """The metaclass for global accessible class.

    The subclasses inheriting from ``ManagerMeta`` will manage their
    own ``_instance_dict`` and root instances. The constructors of subclasses
    must contain the ``name`` argument.

    Examples:
        >>> class SubClass1(metaclass=ManagerMeta):
        >>>     def __init__(self, *args, **kwargs):
        >>>         pass
        AssertionError: <class '__main__.SubClass1'>.__init__ must have the
        name argument.
        >>> class SubClass2(metaclass=ManagerMeta):
        >>>     def __init__(self, name):
        >>>         pass
        >>> # valid format.
    """

    def __init__(cls, *args):
        cls._instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls)
        params_names = params[0] if params[0] else []
        assert 'name' in params_names, f'{cls} must have the `name` argument'
        super().__init__(*args)
class ManagerMixin(metaclass=ManagerMeta):
    """``ManagerMixin`` is the base class for classes that have global access
    requirements.

    The subclasses inheriting from ``ManagerMixin`` can get their
    global instances.

    Examples:
        >>> class GlobalAccessible(ManagerMixin):
        >>>     def __init__(self, name=''):
        >>>         super().__init__(name)
        >>>
        >>> GlobalAccessible.get_instance('name')
        >>> instance_1 = GlobalAccessible.get_instance('name')
        >>> instance_2 = GlobalAccessible.get_instance('name')
        >>> assert id(instance_1) == id(instance_2)

    Args:
        name (str): Name of the instance. Defaults to ''.
    """

    def __init__(self, name: str = '', **kwargs):
        assert isinstance(name, str) and name, \
            'name argument must be an non-empty string.'
        self._instance_name = name

    @classmethod
    def get_instance(cls: Type[T], name: str, **kwargs) -> T:
        """Get subclass instance by name if the name exists.

        If corresponding name instance has not been created, ``get_instance``
        will create an instance, otherwise ``get_instance`` will return the
        corresponding instance.

        Examples
            >>> instance1 = GlobalAccessible.get_instance('name1')
            >>> # Create name1 instance.
            >>> instance.instance_name
            name1
            >>> instance2 = GlobalAccessible.get_instance('name1')
            >>> # Get name1 instance.
            >>> assert id(instance1) == id(instance2)

        Args:
            name (str): Name of instance. Defaults to ''.

        Returns:
            object: Corresponding name instance, the latest instance, or root
            instance.
        """
        _accquire_lock()
        assert isinstance(name, str), \
            f'type of name should be str, but got {type(cls)}'
        instance_dict = cls._instance_dict  # type: ignore
        # Get the instance by name.
        if name not in instance_dict:
            instance = cls(name=name, **kwargs)  # type: ignore
            instance_dict[name] = instance  # type: ignore
        # Get latest instantiated instance or root instance.
        _release_lock()
        return instance_dict[name]

    @classmethod
    def get_current_instance(cls):
        """Get latest created instance.

        Before calling ``get_current_instance``, The subclass must have called
        ``get_instance(xxx)`` at least once.

        Examples
            >>> instance = GlobalAccessible.get_current_instance()
            AssertionError: At least one of name and current needs to be set
            >>> instance = GlobalAccessible.get_instance('name1')
            >>> instance.instance_name
            name1
            >>> instance = GlobalAccessible.get_current_instance()
            >>> instance.instance_name
            name1

        Returns:
            object: Latest created instance.
        """
        _accquire_lock()
        if not cls._instance_dict:
            raise RuntimeError(
                f'Before calling {cls.__name__}.get_current_instance(), you '
                'should call get_instance(name=xxx) at least once.')
        name = next(iter(reversed(cls._instance_dict)))
        _release_lock()
        return cls._instance_dict[name]

    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        """Check whether the name corresponding instance exists.

        Args:
            name (str): Name of instance.

        Returns:
            bool: Whether the name corresponding instance exists.
        """
        return name in cls._instance_dict

    @property
    def instance_name(self) -> str:
        """Get the name of instance.

        Returns:
            str: Name of instance.
        """
        return self._instance_name
class MessageHub(ManagerMixin):
    """Message hub for component interaction. MessageHub is created and
    accessed in the same way as ManagerMixin.

    ``MessageHub`` will record log information and runtime information. The
    log information refers to the learning rate, loss, etc. of the models
    during training phase, which will be stored as ``HistoryBuffer``. The
    runtime information refers to the iter times, meta information of
    runner etc., which will be overwritten by next update.

    Args:
        name (str): Name of message hub used to get corresponding instance
            globally.
        log_scalars (dict, optional): Each key-value pair in the
            dictionary is the name of the log information such as "loss", "lr",
            "metric" and their corresponding values. The type of value must be
            HistoryBuffer. Defaults to None.
        runtime_info (dict, optional): Each key-value pair in the
            dictionary is the name of the runtime information and their
            corresponding values. Defaults to None.
        resumed_keys (dict, optional): Each key-value pair in the
            dictionary decides whether the key in :attr:`_log_scalars` and
            :attr:`_runtime_info` will be serialized.

    Note:
        Key in :attr:`_resumed_keys` belongs to :attr:`_log_scalars` or
        :attr:`_runtime_info`. The corresponding value cannot be set
        repeatedly.

    Examples:
        >>> # create empty `MessageHub`.
        >>> message_hub1 = MessageHub('name')
        >>> log_scalars = dict(loss=HistoryBuffer())
        >>> runtime_info = dict(task='task')
        >>> resumed_keys = dict(loss=True)
        >>> # create `MessageHub` from data.
        >>> message_hub2 = MessageHub(
        >>>     name='name',
        >>>     log_scalars=log_scalars,
        >>>     runtime_info=runtime_info,
        >>>     resumed_keys=resumed_keys)
    """

    def __init__(self,
                 name: str,
                 log_scalars: Optional[dict] = None,
                 runtime_info: Optional[dict] = None,
                 resumed_keys: Optional[dict] = None):
        super().__init__(name)
        self._log_scalars = self._parse_input('log_scalars', log_scalars)
        self._runtime_info = self._parse_input('runtime_info', runtime_info)
        self._resumed_keys = self._parse_input('resumed_keys', resumed_keys)

        for value in self._log_scalars.values():
            assert isinstance(value, HistoryBuffer), \
                ("The type of log_scalars'value must be HistoryBuffer, but "
                 f'got {type(value)}')

        for key in self._resumed_keys.keys():
            assert key in self._log_scalars or key in self._runtime_info, \
                ('Key in `resumed_keys` must contained in `log_scalars` or '
                 f'`runtime_info`, but got {key}')

    @classmethod
    def get_current_instance(cls) -> 'MessageHub':
        """Get latest created ``MessageHub`` instance.

        :obj:`MessageHub` can call :meth:`get_current_instance` before any
        instance has been created, and return a message hub with the instance
        name "mmengine".

        Returns:
            MessageHub: Empty ``MessageHub`` instance.
        """
        if not cls._instance_dict:
            cls.get_instance('mmengine')
        return super().get_current_instance()

    def update_scalar(self,
                      key: str,
                      value: Union[int, float, np.ndarray, 'torch.Tensor'],
                      count: int = 1,
                      resumed: bool = True) -> None:
        """Update :attr:_log_scalars.

        Update ``HistoryBuffer`` in :attr:`_log_scalars`. If corresponding key
        ``HistoryBuffer`` has been created, ``value`` and ``count`` is the
        argument of ``HistoryBuffer.update``, Otherwise, ``update_scalar``
        will create an ``HistoryBuffer`` with value and count via the
        constructor of ``HistoryBuffer``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> # create loss `HistoryBuffer` with value=1, count=1
            >>> message_hub.update_scalar('loss', 1)
            >>> # update loss `HistoryBuffer` with value
            >>> message_hub.update_scalar('loss', 3)
            >>> message_hub.update_scalar('loss', 3, resumed=False)
            AssertionError: loss used to be true, but got false now. resumed
            keys cannot be modified repeatedly'

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``key``.

        Args:
            key (str): Key of ``HistoryBuffer``.
            value (torch.Tensor or np.ndarray or int or float): Value of log.
            count (torch.Tensor or np.ndarray or int or float): Accumulation
                times of log, defaults to 1. `count` will be used in smooth
                statistics.
            resumed (str): Whether the corresponding ``HistoryBuffer``
                could be resumed. Defaults to True.
        """
        self._set_resumed_keys(key, resumed)
        checked_value = self._get_valid_value(value)
        assert isinstance(count, int), (
            f'The type of count must be int. but got {type(count): {count}}')
        if key in self._log_scalars:
            self._log_scalars[key].update(checked_value, count)
        else:
            self._log_scalars[key] = HistoryBuffer([checked_value], [count])

    def update_scalars(self, log_dict: dict, resumed: bool = True) -> None:
        """Update :attr:`_log_scalars` with a dict.

        ``update_scalars`` iterates through each pair of log_dict key-value,
        and calls ``update_scalar``. If type of value is dict, the value should
        be ``dict(value=xxx) or dict(value=xxx, count=xxx)``. Item in
        ``log_dict`` has the same resume option.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``log_dict``.

        Args:
            log_dict (str): Used for batch updating :attr:`_log_scalars`.
            resumed (bool): Whether all ``HistoryBuffer`` referred in
                log_dict should be resumed. Defaults to True.

        Examples:
            >>> message_hub = MessageHub.get_instance('mmengine')
            >>> log_dict = dict(a=1, b=2, c=3)
            >>> message_hub.update_scalars(log_dict)
            >>> # The default count of  `a`, `b` and `c` is 1.
            >>> log_dict = dict(a=1, b=2, c=dict(value=1, count=2))
            >>> message_hub.update_scalars(log_dict)
            >>> # The count of `c` is 2.
        """
        assert isinstance(log_dict, dict), ('`log_dict` must be a dict!, '
                                            f'but got {type(log_dict)}')
        for log_name, log_val in log_dict.items():
            if isinstance(log_val, dict):
                assert 'value' in log_val, \
                    f'value must be defined in {log_val}'
                count = self._get_valid_value(log_val.get('count', 1))
                value = log_val['value']
            else:
                count = 1
                value = log_val
            assert isinstance(count,
                              int), ('The type of count must be int. but got '
                                     f'{type(count): {count}}')
            self.update_scalar(log_name, value, count, resumed)

    def update_info(self, key: str, value: Any, resumed: bool = True) -> None:
        """Update runtime information.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``key``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> message_hub.update_info('iter', 100)

        Args:
            key (str): Key of runtime information.
            value (Any): Value of runtime information.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        self._set_resumed_keys(key, resumed)
        self._runtime_info[key] = value

    def pop_info(self, key: str, default: Optional[Any] = None) -> Any:
        """Remove runtime information by key. If the key does not exist, this
        method will return the default value.

        Args:
            key (str): Key of runtime information.
            default (Any, optional): The default returned value for the
                given key.

        Returns:
            Any: The runtime information if the key exists.
        """
        return self._runtime_info.pop(key, default)

    def update_info_dict(self, info_dict: dict, resumed: bool = True) -> None:
        """Update runtime information with dictionary.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``info_dict``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> message_hub.update_info({'iter': 100})

        Args:
            info_dict (str): Runtime information dictionary.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        assert isinstance(info_dict, dict), ('`log_dict` must be a dict!, '
                                             f'but got {type(info_dict)}')
        for key, value in info_dict.items():
            self.update_info(key, value, resumed=resumed)

    def _set_resumed_keys(self, key: str, resumed: bool) -> None:
        """Set corresponding resumed keys.

        This method is called by ``update_scalar``, ``update_scalars`` and
        ``update_info`` to set the corresponding key is true or false in
        :attr:`_resumed_keys`.

        Args:
            key (str): Key of :attr:`_log_scalrs` or :attr:`_runtime_info`.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        if key not in self._resumed_keys:
            self._resumed_keys[key] = resumed
        else:
            assert self._resumed_keys[key] == resumed, \
                f'{key} used to be {self._resumed_keys[key]}, but got ' \
                '{resumed} now. resumed keys cannot be modified repeatedly.'

    @property
    def log_scalars(self) -> OrderedDict:
        """Get all ``HistoryBuffer`` instances.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will return a reference of
            history buffer rather than a copy.

        Returns:
            OrderedDict: All ``HistoryBuffer`` instances.
        """
        return self._log_scalars

    @property
    def runtime_info(self) -> OrderedDict:
        """Get all runtime information.

        Returns:
            OrderedDict: A copy of all runtime information.
        """
        return self._runtime_info

    def get_scalar(self, key: str) -> HistoryBuffer:
        """Get ``HistoryBuffer`` instance by key.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will not return a reference of
            history buffer rather than a copy.

        Args:
            key (str): Key of ``HistoryBuffer``.

        Returns:
            HistoryBuffer: Corresponding ``HistoryBuffer`` instance if the
            key exists.
        """
        if key not in self.log_scalars:
            raise KeyError(f'{key} is not found in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return self.log_scalars[key]

    def get_info(self, key: str, default: Optional[Any] = None) -> Any:
        """Get runtime information by key. If the key does not exist, this
        method will return default information.

        Args:
            key (str): Key of runtime information.
            default (Any, optional): The default returned value for the
                given key.

        Returns:
            Any: A copy of corresponding runtime information if the key exists.
        """
        if key not in self.runtime_info:
            return default
        else:
            # TODO: There are restrictions on objects that can be saved
            # return copy.deepcopy(self._runtime_info[key])
            return self._runtime_info[key]

    def _get_valid_value(
        self,
        value: Union['torch.Tensor', np.ndarray, np.number, int, float],
    ) -> Union[int, float]:
        """Convert value to python built-in type.

        Args:
            value (torch.Tensor or np.ndarray or np.number or int or float):
                value of log.

        Returns:
            float or int: python built-in type value.
        """
        if isinstance(value, (np.ndarray, np.number)):
            assert value.size == 1
            value = value.item()
        elif isinstance(value, (int, float)):
            value = value
        else:
            # check whether value is torch.Tensor but don't want
            # to import torch in this file
            assert hasattr(value, 'numel') and value.numel() == 1
            value = value.item()
        return value  # type: ignore

    def state_dict(self) -> dict:
        """Returns a dictionary containing log scalars, runtime information and
        resumed keys, which should be resumed.

        The returned ``state_dict`` can be loaded by :meth:`load_state_dict`.

        Returns:
            dict: A dictionary contains ``log_scalars``, ``runtime_info`` and
            ``resumed_keys``.
        """
        saved_scalars = OrderedDict()
        saved_info = OrderedDict()

        for key, value in self._log_scalars.items():
            if self._resumed_keys.get(key, False):
                saved_scalars[key] = copy.deepcopy(value)

        for key, value in self._runtime_info.items():
            if self._resumed_keys.get(key, False):
                try:
                    saved_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    saved_info[key] = value
        return dict(
            log_scalars=saved_scalars,
            runtime_info=saved_info,
            resumed_keys=self._resumed_keys)

    def load_state_dict(self, state_dict: Union['MessageHub', dict]) -> None:
        """Loads log scalars, runtime information and resumed keys from
        ``state_dict`` or ``message_hub``.

        If ``state_dict`` is a dictionary returned by :meth:`state_dict`, it
        will only make copies of data which should be resumed from the source
        ``message_hub``.

        If ``state_dict`` is a ``message_hub`` instance, it will make copies of
        all data from the source message_hub. We suggest to load data from
        ``dict`` rather than a ``MessageHub`` instance.

        Args:
            state_dict (dict or MessageHub): A dictionary contains key
                ``log_scalars`` ``runtime_info`` and ``resumed_keys``, or a
                MessageHub instance.
        """
        if isinstance(state_dict, dict):
            for key in ('log_scalars', 'runtime_info', 'resumed_keys'):
                assert key in state_dict, (
                    'The loaded `state_dict` of `MessageHub` must contain '
                    f'key: `{key}`')
            # The old `MessageHub` could save non-HistoryBuffer `log_scalars`,
            # therefore the loaded `log_scalars` needs to be filtered.
            for key, value in state_dict['log_scalars'].items():
                if not isinstance(value, HistoryBuffer):
                    continue
                self.log_scalars[key] = value

            for key, value in state_dict['runtime_info'].items():
                try:
                    self._runtime_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    self._runtime_info[key] = value

            for key, value in state_dict['resumed_keys'].items():
                if key not in set(self.log_scalars.keys()) | \
                        set(self._runtime_info.keys()):
                    continue
                elif not value:
                    continue
                self._resumed_keys[key] = value

        # Since some checkpoints saved serialized `message_hub` instance,
        # `load_state_dict` support loading `message_hub` instance for
        # compatibility
        else:
            self._log_scalars = copy.deepcopy(state_dict._log_scalars)
            self._runtime_info = copy.deepcopy(state_dict._runtime_info)
            self._resumed_keys = copy.deepcopy(state_dict._resumed_keys)

    def _parse_input(self, name: str, value: Any) -> OrderedDict:
        """Parse input value.

        Args:
            name (str): name of input value.
            value (Any): Input value.

        Returns:
            dict: Parsed input value.
        """
        if value is None:
            return OrderedDict()
        elif isinstance(value, dict):
            return OrderedDict(value)
        else:
            raise TypeError(f'{name} should be a dict or `None`, but '
                            f'got {type(name)}')
class Optimizer(object):
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        torch._C._log_api_usage_once("python.optimizer")
        self.defaults = defaults

        self._hook_for_profile()

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._hook_for_profile()  # To support multiprocessing pickle/unpickle.

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def _hook_for_profile(self):
        self._zero_grad_profile_name = "Optimizer.zero_grad#{}.zero_grad".format(self.__class__.__name__)

        def profile_hook_step(func):

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                obj, *_ = args
                profile_name = "Optimizer.step#{}.step".format(obj.__class__.__name__)
                with torch.autograd.profiler.record_function(profile_name):
                    return func(*args, **kwargs)
            return wrapper

        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = profile_hook_step(self.__class__.step)
            self.__class__.step.hooked = True

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        if not hasattr(self, "_zero_grad_profile_name"):
            self._hook_for_profile()
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            p.grad.zero_()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (callable): A closure that reevaluates the models and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']


        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
class OptimWrapper(BaseOptimWrapper):

    def __init__(self,
                 optimizer: Optimizer,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[dict] = None):
        assert accumulative_counts > 0, (
            '_accumulative_counts at least greater than or equal to 1')
        self._accumulative_counts = accumulative_counts
        self.optimizer = optimizer

        if clip_grad is not None:
            # clip_grad_kwargs should not be non-empty dict.
            assert isinstance(clip_grad, dict) and clip_grad, (
                'If `clip_grad` is not None, it should be a `dict` '
                'which is the arguments of `torch.nn.utils.clip_grad_norm_` '
                'or clip_grad_value_`.')
            clip_type = clip_grad.pop('type', 'norm')
            if clip_type == 'norm':
                self.clip_func = torch.nn.utils.clip_grad_norm_
                self.grad_name = 'grad_norm'
            elif clip_type == 'value':
                self.clip_func = torch.nn.utils.clip_grad_value_
                self.grad_name = 'grad_value'
            else:
                raise ValueError('type of clip_grad should be "norm" or '
                                 f'"value" but got {clip_type}')
            assert clip_grad, ('`clip_grad` should contain other arguments '
                               'besides `type`. The arguments should match '
                               'with the `torch.nn.utils.clip_grad_norm_` or '
                               'clip_grad_value_`')
        self.clip_grad_kwargs = clip_grad
        # Used to update `grad_norm` log message.
        self.message_hub = MessageHub.get_current_instance()
        self._inner_count = 0
        # `_max_counts` means the total number of parameter updates.  It
        # ensures that the gradient of the last few iterations will not be
        # lost when the `_max_counts` is not divisible by
        # `accumulative_counts`.
        self._max_counts = -1
        # The `_remainder_iter` is used for calculating loss factor at the
        # last few iterations. If `_max_counts` has not been initialized,
        # the loss factor will always be the same as `_accumulative_counts`.
        self._remainder_counts = -1

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

    def update_params(  # type: ignore
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation.

        Provide unified ``backward`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on GradScaler during backward process.

        Note:
            If subclasses inherit from ``OptimWrapper`` override
            ``backward``, ``_inner_count +=1`` must be implemented.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`.
        """
        loss.backward(**kwargs)
        self._inner_count += 1

    def zero_grad(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.zero_grad``.

        Provide unified ``zero_grad`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.zero_grad`.
        """
        self.optimizer.zero_grad(**kwargs)

    def step(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.step``.

        Provide unified ``step`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on ``GradScaler`` during step process.

        Clip grad if :attr:`clip_grad_kwargs` is not None, and then update
        parameters.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.
        """
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.step(**kwargs)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """A Context for gradient accumulation and automatic mix precision
        training.

        If subclasses need to enable the context for mix precision training,
        e.g., ``:class:`AmpOptimWrapper``,  the corresponding context should be
        enabled in `optim_context`. Since ``OptimWrapper`` uses default fp32
        training, ``optim_context`` will only enable the context for
        blocking the unnecessary gradient synchronization during gradient
        accumulation

        If models is an instance with ``no_sync`` method (which means
        blocking the gradient synchronization) and
        ``self._accumulative_counts != 1``. The models will not automatically
        synchronize gradients if ``cur_iter`` is divisible by
        ``self._accumulative_counts``. Otherwise, this method will enable an
        empty context.

        Args:
            model (nn.Module): The training models.
        """
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        if not self.should_sync() and hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield

    def _clip_grad(self) -> None:
        """Clip the gradients of parameters."""
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group['params'])

        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad = self.clip_func(params, **self.clip_grad_kwargs)
            # `torch.nn.utils.clip_grad_value_` will return None.
            if grad is not None:
                self.message_hub.update_scalar(f'train/{self.grad_name}',
                                               float(grad))

    def initialize_count_status(self, model: nn.Module, init_counts: int,
                                max_counts: int) -> None:
        """Initialize gradient accumulation related attributes.

        ``OptimWrapper`` can be used without calling
        ``initialize_iter_status``. However, Consider the case of  ``len(
        dataloader) == 10``, and the ``accumulative_iter == 3``. Since 10 is
        not divisible by 3, the last iteration will not trigger
        ``optimizer.step()``, resulting in one less parameter updating.

        Args:
            model (nn.Module): Training models
            init_counts (int): The initial value of the inner count.
            max_counts (int): The maximum value of the inner count.
        """
        self._inner_count = init_counts
        self._max_counts = max_counts

        if has_batch_norm(model) and self._accumulative_counts > 1:
            pass
        # Remainder of `_max_counts` divided by `_accumulative_counts`
        self._remainder_counts = self._max_counts % self._accumulative_counts

    def should_update(self) -> bool:
        """Decide whether the parameters should be updated at the current
        iteration.

        Called by :meth:`update_params` and check whether the optimizer
        wrapper should update parameters at current iteration.

        Returns:
            bool: Whether to update parameters.
        """
        return (self._inner_count % self._accumulative_counts == 0
                or self._inner_count == self._max_counts)

    def should_sync(self) -> bool:
        """Decide whether the automatic gradient synchronization should be
        allowed at the current iteration.

        It takes effect when gradient accumulation is used to skip
        synchronization at the iterations where the parameter is not updated.

        Since ``should_sync`` is called by :meth:`optim_context`, and it is
        called before :meth:`backward` which means ``self._inner_count += 1``
        has not happened yet. Therefore, ``self._inner_count += 1`` should be
        performed manually here.

        Returns:
            bool: Whether to block the automatic gradient synchronization.
        """
        return ((self._inner_count + 1) % self._accumulative_counts == 0
                or (self._inner_count + 1) == self._max_counts)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Get scaled loss according to ``_accumulative_counts``,
        ``_inner_count`` and max_counts.

        Args:
            loss (torch.Tensor): Original loss calculated by models.

        Returns:
            loss (torch.Tensor): Scaled loss.
        """
        if self._accumulative_counts == 1:
            # update parameters without gradient accumulation. The gradient
            # should not be rescaled and `loss_factor=1`.
            loss_factor = 1
        elif self._max_counts == -1:
            loss_factor = self._accumulative_counts
        else:
            # if `self._accumulative_counts > 1`, the gradient needs to be
            # rescaled and accumulated. In most cases, `loss_factor` equals to
            # `self._accumulative_counts`. However, `self._max_counts` may not
            # be divisible by `self._accumulative_counts`, so the
            # `loss_scale` for the last few iterations needs to be
            # recalculated.
            if self._inner_count < self._max_counts - self._remainder_counts:
                loss_factor = self._accumulative_counts
            else:
                loss_factor = self._remainder_counts
            assert loss_factor > 0, (
                'loss_factor should be larger than zero! This error could '
                'happened when initialize_iter_status called with an '
                'error `init_counts` or `max_counts`')

        loss = loss / loss_factor
        return loss

    @property
    def inner_count(self):
        """Get the number of updating parameters of optimizer wrapper."""
        return self._inner_count

    def __repr__(self):
        wrapper_info = (f'Type: {type(self).__name__}\n'
                        f'_accumulative_counts: {self._accumulative_counts}\n'
                        'optimizer: \n')
        optimizer_str = repr(self.optimizer) + '\n'
        return wrapper_info + optimizer_str