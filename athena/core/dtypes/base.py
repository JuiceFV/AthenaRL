from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torchrec import KeyedJaggedTensor, PoolingType

from athena.core.base_dclass import BaseDataClass
from athena.core.dataclasses import dataclass as pydantic_dataclass
from athena.core.enum_meta import AthenaEnumMeta, Enum
from athena.core.logger import LoggerMixin


@dataclass
class TensorDataClass(BaseDataClass, LoggerMixin):
    r"""
    The base data structure represents n-dimensional tensor-based data.
    Generally, we don't need the internal :class:`torch.Tensor` implementation
    to represent tensor-based data, i.e. the explicit interface is enough.
    If a structure has multiple :class:`torch.Tensor` fields then an attribute
    call will be applied to each one.

    Example::

        @dataclass
        class DocSeq(TensorDataClass):
            dense_features: torch.Tensor
            mask: Optional[torch.Tensor] = None

        docs = DocSeq(torch.Tensor(1, 3), torch.ones(1,3, dtype=torch.bool))
        docs.is_shared() # DocSeq(dense_features=False, mask=False)
    """

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
            return type(self)(**recursive_call(self.__dict__))
        return tensor_attrs_call

    def cuda(self, *args, **kwargs) -> Union["TensorDataClass", torch.Tensor]:
        """Returns a copy of this object in CUDA memory.

        Args:
            *args: Arguments required by :func:`torch.Tensor.cuda`
            **kwargs: Keyword arguments required by :func:`torch.Tensor.cuda`

        Returns:
            Union["TensorDataClass", torch.Tensor]: Copy of the object
        """
        cuda_tensor = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                kwargs["non_blocking"] = kwargs.get("non_blocking", True)
                cuda_tensor[k] = v.cuda(*args, **kwargs)
            elif isinstance(v, TensorDataClass):
                cuda_tensor[k] = v.cuda(*args, **kwargs)
            else:
                cuda_tensor[k] = v
        return type(self)(**cuda_tensor)

    def cpu(self) -> Union["TensorDataClass", torch.Tensor]:
        r"""
        Returns a copy of this object in CPU memory.

        Returns:
            Union["TensorDataClass", torch.Tensor]: Copy of the object.
        """
        cpu_tensor = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (torch.Tensor, TensorDataClass)):
                cpu_tensor[k] = v.cpu()
            else:
                cpu_tensor[k] = v
        return type(self)(**cpu_tensor)


@dataclass
class Feature(TensorDataClass):
    r"""
    Feature wrapper which helps to handle different types of features.

    .. warning::

        Currently, only dense features are in use. Thus, there is no implementation
        of catigorical features.
    """

    #: Dense float features. (E.g. time spent)
    dense_features: torch.Tensor

    #: Categorical features. (E.g. Page ID)
    categorical_features: Optional[KeyedJaggedTensor] = None

    #: Ctaegorical features which is measured somehow. (E.g. Page time creation)
    scored_categorical_features: Optional[KeyedJaggedTensor] = None


class Ftype(str, Enum, metaclass=AthenaEnumMeta):
    r"""
    Feature type which is detected while Data Analysis
    process.
    """

    #: Feature value can be either binary (0 or 1) or unique (``min == max``).
    BINARY = "binary"

    #: Feature value is a real number and a distribution adheres normal one.
    CONTINUOUS = "continuous"

    #: Feature value lies within range :math:`[0; 1]`.
    PROBABILITY = "probability"

    #: Feature value is a real number whose distribution differs enough from
    #: normal to apply the box-cox transformation.
    BOXCOX = "boxcox"

    #: Feature takes any discrete value which will be processed distinctly.
    ENUM = "enum"

    #: Feature value is a real number whose distribution differs enough from
    #: normal to apply the quantile normalization.
    QUANTILE = "quantile"

    #: Feature will not be processed. Commonly used for fake features.
    DO_NOT_PREPROCESS = "do_not_preprocess"


@pydantic_dataclass
class ContinuousFeatureInfo(BaseDataClass):
    r"""
    Float feature information used in :class:`ModelFeatureConfig`.
    """

    #: Feature name.
    name: str

    #: Unique feature id.
    feature_id: int


@pydantic_dataclass
class IDListFeatureConfig(BaseDataClass):
    r"""
    Categorical feature information used in :class:`ModelFeatureConfig`.

    .. important::

        :class:`KeyedJaggedTensor` required to store this type of feature.
    """

    #: Feature name.
    name: str

    #: Unique feature id.
    feature_id: int

    #: Name of the embedding table to use.
    id_mapping_name: str


@pydantic_dataclass
class IDScoreListFeatureConfig(BaseDataClass):
    r"""
    Additional cstegorical feature information used in :class:`ModelFeatureConfig`.
    """

    #: Feature name.
    name: str

    #: Unique feature id.
    feature_id: int

    #: Name of the embedding table to use.
    id_mapping_name: str


@pydantic_dataclass
class IDMappingConfig:
    r"""
    Configuration of an embedding table used to store categorical features.
    Multiple features may share the same embedding table.
    """

    #: Embedding table size.
    embedding_table_size: int

    #: Output embedding dimensions.
    embedding_dim: int

    #: Whether to perform hashing to make id fall in the range of embedding_table_size.
    #: If False, the user is at their own risk of raw ids going beyond the range.
    hashing: bool = True

    #: Type of pooling while map reduce.
    pooling_type: PoolingType = PoolingType.MEAN

    def __eq__(self, __o: "IDMappingConfig") -> bool:
        return (
            self.embedding_table_size == __o.embedding_table_size
            and self.embedding_dim == __o.embedding_dim
            and self.hashing == __o.hashing
            and self.pooling_type == __o.pooling_type
        )


@pydantic_dataclass
class ModelFeatureConfig(BaseDataClass):
    r"""
    Model Feature Configuration - data understanding tool.
    All features are stored in DWH according to a given schema.

    +---------------------+------------------------+-------------------------------+
    |      Continuous     |         ID List        |         ID Score List         |
    +=====================+========================+===============================+
    | ``map<int, float>`` | ``map<int, list[int]>``| ``map<int, map<int, float>>`` |
    +---------------------+------------------------+-------------------------------+
    | ...                 | ...                    | ...                           |
    +---------------------+------------------------+-------------------------------+

    To ensure interoperability of data across multiple models we store two types of features,
    dense and sparse.

    1. Dense continuous feature - maps fearue id to a float feature value. (E.g. time on page)
    2. Sparse ID List - maps feature id to a catigorical value, commonly one-hot. (E.g. page id)
    3. Sparse ID Score List - associates each categorical value with some "score" (E.g. page creation time)

    To properly treat a ton of features during analysis this dataclass helps
    a user to map feature configuration (e.g. feature id, name, embedding info etc.)
    """

    #: Continuous feature info.
    continuous_feature_infos: List[ContinuousFeatureInfo] = field(default_factory=list)
    #: Configuration of partitions, represented as mapping between partition name and its configuration.
    id_mapping_config: Dict[str, IDMappingConfig] = field(default_factory=dict)
    #: Categorical feature info.
    id_list_feature_configs: List[IDListFeatureConfig] = field(default_factory=list)
    #: Scored categorical feature info.
    id_score_list_feature_configs: List[IDScoreListFeatureConfig] = field(default_factory=list)

    def __post_init_post_parse__(self):
        """Sanity checks if sparse features were given.

        Raises:
            RuntimeError: If feature ids contain duplicates.
            RuntimeError: If feature names contain duplicates.
            RuntimeError: If number of feature ids is not the same as number of feature names.
            RuntimeError: If feature mappings doesn't match with mapping configs.
        """
        sparse_features = self.id_list_feature_configs + self.id_score_list_feature_configs
        if not self.dense_only:
            fids = [config.feature_id for config in sparse_features]
            names = [config.name for config in sparse_features]
            if len(fids) != len(set(fids)):
                raise RuntimeError(f"Duplicates in fids: {fids}")
            if len(names) != len(set(names)):
                raise RuntimeError(f"Duplicates in names: {names}")
            if len(fids) != len(names):
                raise RuntimeError("Incosistent lengths of mapping fid -> name.")
            id_mapping_names = [config.id_mapping_name for config in sparse_features]
            if set(id_mapping_names) != set(self.id_mapping_config.keys()):
                raise RuntimeError(
                    f"id_mapping_names in id_list_feature_configs/id_score_list_feature_configs "
                    f"({set(id_mapping_names)}) not match with those in "
                    f"id_mapping_config ({set(self.id_mapping_config.keys())})"
                )
        self._fid2name = {config.feature_id: config.name for config in sparse_features}
        self._name2fid = {config.name: config.feature_id for config in sparse_features}
        self._fid2config = {config.feature_id: config for config in sparse_features}
        self._name2config = {config.name: config for config in sparse_features}

    @property
    def dense_only(self) -> bool:
        r"""
        If only dense features defined.

        Returns:
            bool: If only dense features.
        """
        return not (self.id_list_feature_configs or self.id_score_list_feature_configs)

    @property
    def fid2name(self) -> Dict[int, str]:
        r"""
        Mapping of feature id and its name.

        Returns:
            Dict[int, str]: Mapping of feature id and its name.
        """
        return self._fid2name

    @property
    def name2fid(self) -> Dict[str, int]:
        r"""
        Mapping of feature name and its id.

        Returns:
            Dict[str, int]: Mapping of feature name and its id.
        """
        return self._name2fid

    @property
    def fid2config(self) -> Dict[int, Union[IDListFeatureConfig, IDScoreListFeatureConfig]]:
        r"""
        Mapping of feature id and its config.

        Returns:
            Dict[int, Union[IDListFeatureConfig, IDScoreListFeatureConfig]]: Mapping of feature id and its config.
        """
        return self._fid2config

    @property
    def name2config(self) -> Dict[str, Union[IDListFeatureConfig, IDScoreListFeatureConfig]]:
        r"""
        Mapping of feature name and its config.

        Returns:
            Dict[str, Union[IDListFeatureConfig, IDScoreListFeatureConfig]]: Mapping of feature name and its config.
        """
        return self._name2config
