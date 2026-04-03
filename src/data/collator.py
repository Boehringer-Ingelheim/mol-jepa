from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch
from torch import Tensor
from torch_geometric.data.storage import NodeStorage
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame
from torch_geometric.data.collate import (
    _collate,
    _batch_and_ptr,
    repeat_interleave,
    cumsum,
)

from collections import defaultdict
from typing import (
    Dict,
    Iterable,
    Tuple,
    Type,
    TypeVar,
)


T = TypeVar("T")
SliceDictType = Dict[str, Union[Tensor, Dict[str, Tensor]]]
IncDictType = Dict[str, Union[Tensor, Dict[str, Tensor]]]


def custom_collate(
    cls: Type[T],
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
    include_keys: Optional[Iterable[str]] = None,
) -> Tuple[T, SliceDictType, IncDictType]:
    if not isinstance(data_list, (list, tuple)):
        data_list = list(data_list)

    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)
    else:
        out = cls()

    out.stores_as(data_list[0])

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])
    include_keys = set(include_keys or [])

    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    device: Optional[torch.device] = None
    slice_dict: SliceDictType = {}
    inc_dict: IncDictType = {}
    for out_store in out.stores:
        key = out_store._key
        stores = key_to_stores[key]

        # Collect attrs across ALL stores, seeded with any required include_keys
        all_attrs: set[str] = set(include_keys)
        for store in stores:
            all_attrs.update(store.keys())

        for attr in all_attrs:
            if attr in exclude_keys:
                continue

            # Per-store: use value if present, else None as a sentinel
            values = [store[attr] if attr in store else None for store in stores]

            # num_nodes needs special treatment - sum values instead of merging
            if attr == "num_nodes":
                resolved = [v if v is not None else 0 for v in values]
                out_store._num_nodes = resolved
                out_store.num_nodes = sum(resolved)
                continue

            # Skip batching of ptr vectors
            if attr == "ptr":
                continue

            # Filter to only the stores/values that actually have this attr
            present_stores = [s for s, v in zip(stores, values) if v is not None]
            present_values = [v for v in values if v is not None]

            # Attribute is required but missing from every store — register
            # zero-width slices and zero increments so downstream code that
            # expects the key to exist in slice_dict/inc_dict won't KeyError.
            if not present_values:
                if attr in include_keys:
                    zero_slices = torch.zeros(len(stores) + 1, dtype=torch.long)
                    zero_incs = torch.zeros(len(stores), dtype=torch.long)
                    if key is not None:
                        slice_dict.setdefault(key, {})[attr] = zero_slices
                        inc_dict.setdefault(key, {})[attr] = zero_incs
                    else:
                        slice_dict[attr] = zero_slices
                        inc_dict[attr] = zero_incs
                # else: fully absent and not required — skip entirely
                continue

            # Collate attributes into a unified representation
            value, present_slices, incs = _collate(
                attr, present_values, data_list, present_stores, increment
            )

            # If parts of the data are already on GPU, make sure that auxiliary
            # data like `batch` or `ptr` are also created on GPU
            if isinstance(value, Tensor) and value.is_cuda:
                device = value.device

            out_store[attr] = value

            # Re-expand slices to cover ALL stores, inserting zero-width
            # slice entries for stores that were missing this attribute.
            # e.g. [0, 3, 3, 7] means item-0 has 3 elems, item-1 has 0, item-2 has 4.
            if isinstance(present_slices, Tensor):
                widths = present_slices[1:] - present_slices[:-1]
                present_iter = iter(widths.tolist())
                full_widths = [
                    next(present_iter) if v is not None else 0 for v in values
                ]
                full_slices = torch.tensor(
                    [0] + list(torch.tensor(full_widths).cumsum(0).tolist()),
                    dtype=present_slices.dtype,
                )
            else:
                widths = [
                    present_slices[i + 1] - present_slices[i]
                    for i in range(len(present_slices) - 1)
                ]
                present_iter = iter(widths)
                full_widths = [
                    next(present_iter) if v is not None else 0 for v in values
                ]
                cumsum_ = 0
                full_slices = [0]
                for w in full_widths:
                    cumsum_ += w
                    full_slices.append(cumsum_)

            if key is not None:  # Heterogeneous
                slice_dict.setdefault(key, {})[attr] = full_slices
                inc_dict.setdefault(key, {})[attr] = incs
            else:  # Homogeneous
                slice_dict[attr] = full_slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute
            if attr in follow_batch:
                batch, ptr = _batch_and_ptr(full_slices, device)
                out_store[f"{attr}_batch"] = batch
                out_store[f"{attr}_ptr"] = ptr

        # In case of node-level storages, add a top-level batch vector
        if (
            add_batch
            and isinstance(stores[0], NodeStorage)
            and stores[0].can_infer_num_nodes
        ):
            repeats = [store.num_nodes or 0 for store in stores]
            out_store.batch = repeat_interleave(repeats, device=device)
            out_store.ptr = cumsum(torch.tensor(repeats, device=device))

    return out, slice_dict, inc_dict


def from_data_list(
    cls,
    data_list: List[BaseData],
    follow_batch: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    include_keys: Optional[List[str]] = None,
):
    r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
    list of :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` objects.
    The assignment vector :obj:`batch` is created on the fly.
    In addition, creates assignment vectors for each key in
    :obj:`follow_batch`.
    Will exclude any keys given in :obj:`exclude_keys`.
    """
    batch, slice_dict, inc_dict = custom_collate(
        cls,
        data_list=data_list,
        increment=True,
        add_batch=not isinstance(data_list[0], Batch),
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
        include_keys=include_keys,
    )

    batch._num_graphs = len(data_list)  # type: ignore
    batch._slice_dict = slice_dict  # type: ignore
    batch._inc_dict = inc_dict  # type: ignore

    return batch


class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        include_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.include_keys = include_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return from_data_list(
                Batch,
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
                include_keys=self.include_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")
