"""Type aliases and definitions used across the project."""

from typing import Union, Optional, Tuple, List, Dict, Any, Callable, Iterator, Sequence, Set
import torch
import numpy as np

Tensor = torch.Tensor
Shape = Tuple[int, ...]
DType = torch.dtype
Device = torch.device

IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor
BoolTensor = torch.BoolTensor

NumpyArray = np.ndarray
TokenIds = List[int]
TextString = str
BatchDict = Dict[str, Tensor]
CheckpointState = Dict[str, Any]
LogData = Dict[str, float]

SEQUENCE = Sequence
MAPPING = Dict
ITERABLE = Iterator
OPTIONAL = Optional
UNION = Union
CALLABLE = Callable
TUPLE = Tuple
LIST = List
SET = Set
ANY = Any
