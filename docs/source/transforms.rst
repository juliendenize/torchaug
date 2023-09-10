Transforms
==========

.. currentmodule:: torchaug.transforms

In this section are defined the classes and functions to transform an image or video.

.. note:: Whilst some of these augmentations accept batched data, it will only sample random parameters once which might be detrimental to an effective learning.

.. note:: Some transforms rely on Torchvision functional.

Wrappers and Bases
------------------

List of wrappers of the data augmentations.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Wrapper
    ImageWrapper
    VideoWrapper

List of bases for the data augmentations and wrappers.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    VideoBase

Transforms
----------

List of the data augmentations.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Div255
    MixUp
    Mul255
    Normalize
    RandomApply
    RandomColorJitter
    RandomGaussianBlur
    RandomSolarize
    RandomTransform
    VideoNormalize


Functional
----------

.. currentmodule:: torchaug.transforms.functional

List of the functions to make the data augmentations.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    adjust_hue
    div_255
    gaussian_blur
    mixup
    mul_255
    normalize
    solarize