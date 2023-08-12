Batch Transforms
================

.. currentmodule:: torchaug.batch_transforms



In this section are defined the classes and functions to transform a batch of images or videos.

.. note:: Whilst these augmentations accept batched data, some of them offer a trade-off between randomness and computational cost such as :py:class:`BatchRandomColorJitter`.

.. note:: Some transforms rely on Torchvision functional.

Transforms
----------

List of class wrappers of the batched data augmentations.


.. autosummary::
    :toctree: generated/
    :template: class.rst

    BatchMixUp
    BatchRandomApply
    BatchRandomColorJitter
    BatchRandomGaussianBlur
    BatchRandomGrayScale
    BatchRandomHorizontalFlip
    BatchRandomResizedCrop
    BatchRandomSolarize
    BatchRandomTransform
    BatchVideoWrapper
    BatchVideoResize


Functional
----------

.. currentmodule:: torchaug.batch_transforms.functional

List of the functions to make the batched data augmentations.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    batch_adjust_brightness
    batch_adjust_contrast
    batch_adjust_hue
    batch_adjust_saturation
    batch_gaussian_blur
    batch_mixup
