"""Basic data types required for the ranking, re-ranking and 
recommendation problem.
"""
import dataclasses
import torch

from dataclasses import dataclass, field
from typing import Any, Optional
from athena.core.base_dclass import BaseDataClass
from athena.core.logger import LoggerMixin


@dataclass
class Dataset:
    json_url: str


@dataclass
class TableSpec:
    table_name: str
    table_sample: Optional[float] = None
    val_table_sample: Optional[float] = None
    test_table_sample: Optional[float] = None


@dataclass
class ReaderOptions:
    minibatch_size: int = 1024
    reader_pool_type: str = "thread"


@dataclass
class TensorDataClass(BaseDataClass, LoggerMixin):
    def __getattr__(self, __name: str):
        if __name.startswith("__") and __name.endswith("__"):
            raise AttributeError(
                "We don't wanna call superprivate method of torch.Tensor"
            )
        tensor_attr = getattr(torch.Tensor, __name, None)

        if tensor_attr is None or not callable(tensor_attr):
            self.error(
                f"Attempting to call {self.__class__.__name__}.{__name} on "
                f"{type(self)} (instance of TensorDataClass)."
            )
            if tensor_attr is None:
                raise AttributeError(
                    f"{self.__class__.__name__} doesn't have {__name} attribute."
                )
            else:
                raise RuntimeError(
                    f"{self.__class__.__name__}.{__name} is not callable."
                )

        def tensor_attrs_call(*args, **kwargs):
            """The TensorDataClass is the base one, thus we wanna get 
            attribute (when we call `__getattr__`) at every single 
            child's `Callable` attribute where it possible (if 
            child's attribute has torch.Tensor instance).
            """
            def recursive_call(obj: Any):
                if isinstance(obj, (torch.Tensor, TensorDataClass)):
                    return getattr(obj, __name)(*args, **kwargs)
                if isinstance(obj, dict):
                    return {key: recursive_call(value) for key, value in obj.items()}
                if isinstance(obj, tuple):
                    return tuple(recursive_call(value) for value in obj)
                return obj
            return type(self)(*recursive_call(**self.__dict__))
        return tensor_attrs_call

    def cuda(self, *args, **kwargs):
        cuda_tensor = {}
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                kwargs["non_blocking"] = kwargs.get("non_blocking", True)
                cuda_tensor[attr_name] = attr_value.cuda(*args, **kwargs)
            elif isinstance(attr_value, TensorDataClass):
                cuda_tensor[attr_name] = attr_value.cuda(*args, **kwargs)
            else:
                cuda_tensor[attr_name] = attr_value
        return type(self)(**cuda_tensor)

    def cpu(self):
        cpu_tensor = {}
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, (torch.Tensor, TensorDataClass)):
                cpu_tensor[attr_name] = attr_value.cpu()
            else:
                cpu_tensor[attr_name] = attr_value
        return type(self)(**cpu_tensor)


@dataclass
class DocSeq(TensorDataClass):
    # Documents feature-wise represented
    # torch.Size([batch_size, num_candidates, num_doc_features])
    repr: torch.Tensor
    mask: torch.Tensor = None
    gain: torch.Tensor = None

    def __post_init__(self):
        if len(self.repr.shape) != 3:
            raise ValueError(
                f"Unexpected shape: {self.repr.shape}"
            )
        if self.mask is None:
            self.mask = self.repr.new_ones(
                self.repr.shape[:2], dtype=torch.bool
            )
        if self.gain is None:
            self.gain = self.repr.new_ones(
                self.repr.shape[:2]
            )


@dataclass
class Feature(TensorDataClass):
    repr: torch.Tensor


@dataclass
class PreprocessedRankingInput(TensorDataClass):
    # latent memory state, i.e. the transduction
    # {x_i}^n -> {e_i}^n, where e_i = Encoder(x_i).
    # In ranking seq2slate problem latent state
    # depicts the items placed at 0 ... t-1 timestamp.
    latent_state: Feature

    # Sequence feature-wise representation {x_i}^n.
    source_seq: Feature

    # Mask for source sequences' items, especially
    # required for the reward (i.e. RL mode) models.
    # It defines the items set to which an item
    # should pay attention to in purpose to increase
    # reward metrics.
    source2source_mask: Optional[torch.Tensor] = None

    # Target sequence passed to the decoder.
    # Used in weak supervised and teacher forcing learning.
    target_input_seq: Optional[Feature] = None

    # Target sequence after passing throughout the decoder
    # and stacked fully-connected layers.
    target_output_seq: Optional[Feature] = None

    # Mask for target sequences' items, s.t.
    # each item of a sequence has its own set of
    # items to pay attention to.
    target2target_mask: Optional[torch.Tensor] = None
    slate_reward: Optional[torch.Tensor] = None
    position_reward: Optional[torch.Tensor] = None

    # all indices will be shifted onto 2 due to padding items
    # start/end symbol.
    source_input_indcs: Optional[torch.Tensor] = None
    target_input_indcs: Optional[torch.Tensor] = None
    target_output_indcs: Optional[torch.Tensor] = None
    target_output_probas: Optional[torch.Tensor] = None

    # Ground-truth target sequence (for teacher forcing)
    gt_target_input_indcs: Optional[torch.Tensor] = None
    gt_target_output_indcs: Optional[torch.Tensor] = None
    gt_target_input_seq: Optional[Feature] = None
    gt_target_output_seq: Optional[Feature] = None

    @property
    def batch_size(self) -> int:
        return self.latent_state.repr.size()[0]

    def __len__(self) -> int:
        return self.batch_size


@dataclass
class RankingOutput(TensorDataClass):
    ordered_target_out_idcs: Optional[torch.Tensor] = None
    ordered_per_item_probas: Optional[torch.Tensor] = None
    ordered_per_seq_probas: Optional[torch.Tensor] = None
    log_probas: Optional[torch.Tensor] = None
    encoder_scores: Optional[torch.Tensor] = None
