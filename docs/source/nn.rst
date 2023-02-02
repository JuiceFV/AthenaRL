.. role:: hidden
    :class: hidden-section

athena.nn
=========
.. automodule:: athena.nn

.. contents:: athena.nn
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: athena

Samplers
--------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.SimplexSampler

Sparse Layers
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Embedding
    nn.TransformerEmbedding

TransformerLayers
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.PTEncoder
    nn.PTDecoder
    nn.PointwisePTDecoderLayer
    nn.PointwisePTDecoder

Positioning
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst
        
    nn.VLPositionalEncoding

Utilities
---------

.. automodule:: athena.nn.utils
.. currentmodule:: athena.nn.utils

From the ``athena.nn.utils`` module

Transformer utilities
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    transformer.subsequent_mask
    transformer.decoder_mask
    transformer.mask_by_index