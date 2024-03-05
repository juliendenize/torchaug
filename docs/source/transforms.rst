Transforms
==========

.. currentmodule:: torchaug.transforms

In this section are defined the classes to transform an image or a video. 

.. note:: Lots of transforms are based on `Torchvision transforms v2 <https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py>`_


Augment
----------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    CutMix
    MixUp
    RandomErasing

Auto-Augment
-------------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    AugMix
    AutoAugment
    RandAugment
    TrivialAugmentWide

Color
----------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    ColorJitter
    Grayscale
    RandomAdjustSharpness
    RandomAutocontrast
    RandomChannelPermutation
    RandomColorJitter
    RandomEqualize
    RandomGrayscale
    RandomInvert
    RandomPhotometricDistort
    RandomPosterize
    RandomSolarize

Container
----------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    Compose
    RandomApply
    RandomChoice
    RandomOrder
    SequentialTransform

Geometry
----------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    CenterCrop
    ElasticTransform
    FiveCrop
    Pad
    RandomAffine
    RandomCrop
    RandomHorizontalFlip
    RandomIoUCrop
    RandomPerspective
    RandomResize
    RandomResizedCrop
    RandomRotation
    RandomShortestSize
    RandomVerticalFlip
    RandomZoomOut
    Resize
    ScaleJitter
    TenCrop

Meta
----------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    ClampBoundingBoxes
    ConvertBoundingBoxFormat

Misc
------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    GaussianBlur
    Identity
    Lambda
    LinearTransformation
    Normalize
    RandomGaussianBlur
    SanitizeBoundingBoxes
    ToDtype

Temporal
--------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    UniformTemporalSubsample

Type Conversion 
----------------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    ToBatchImages
    ToImage
    ToPureTensor
