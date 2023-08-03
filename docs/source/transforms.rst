Transforms
==========

.. currentmodule:: torchaug.transforms

In this section are defined the classes and functions to transform an image or video.

.. note:: Whilst some of these augmentations accept batched data, it will only sample random parameters once which might be detrimental to an effective learning.

.. note:: Some transforms rely on Torchvision functional.

Transforms
----------

List of class wrappers of the data augmentations.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Div255
    Mul255
    Normalize
    RandomApply
    RandomColorJitter
    RandomGaussianBlur
    RandomSolarize
    RandomTransform
    VideoNormalize
    VideoWrapper


Functional
----------

.. currentmodule:: torchaug.transforms.functional

List of the functions to make the data augmentations.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    div_255
    gaussian_blur
    mul_255
    normalize
    solarize
