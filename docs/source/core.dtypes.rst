.. role:: hidden
    :class: hidden-section

athena.core.dtypes
==================
.. automodule:: athena.core.dtypes
.. currentmodule:: athena.core.dtypes

.. contents:: athena.core.dtypes
    :depth: 2
    :local:
    :backlinks: top

Base
----

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TensorDataClass
    Ftype
    ModelFeatureConfig
    IDMappingConfig
    IDScoreListFeatureConfig
    IDListFeatureConfig
    Feature

Dataset
-------

.. currentmodule:: athena.core.dtypes.dataset

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Dataset
    TableSpec

Preprocessing
-------------

.. currentmodule:: athena.core.dtypes.preprocessing

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    InputColumn

Ranking
-------

.. currentmodule:: athena.core.dtypes.ranking

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    PreprocessedRankingInput
    RankingOutput
    ~seq2slate.Seq2SlateTransformerOutput
    ~seq2slate.Seq2SlateOutputArch
    ~seq2slate.Seq2SlateMode
    ~seq2slate.Seq2SlateVersion

Reinforcement Learning
----------------------

.. currentmodule:: athena.core.dtypes.rl

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    IPSBlurMethod
    IPSBlur
    RLOptions
    RewardOptions
    ExtraData

Options
-------

.. currentmodule:: athena.core.dtypes

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~rl.RLOptions
    ~rl.RewardOptions
    ~preprocessing.PreprocessingOptions
    ~options.ReaderOptions
    ~options.ResourceOptions
    ~options.AthenaOptions
