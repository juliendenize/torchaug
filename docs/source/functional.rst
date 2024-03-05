Functional
==========

.. currentmodule:: torchaug.transforms.functional

In this section are defined the functions to transform an image or a video.

.. note:: Some transforms rely on `Torchvision transforms v2 functionals  <https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py>`_



Augment
-------

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    erase
    erase_image
    erase_video

Color
----------
.. autosummary::
    :toctree: generated/
    :template: function.rst

    adjust_brightness
    adjust_brightness_batch
    adjust_contrast
    adjust_contrast_batch
    adjust_hue
    adjust_hue_batch
    adjust_saturation
    adjust_saturation_batch
    adjust_sharpness
    autocontrast
    equalize
    invert
    permute_channels
    posterize
    rgb_to_grayscale
    solarize


Geometry
----------
.. autosummary::
    :toctree: generated/
    :template: function.rst

    affine
    center_crop
    crop
    elastic
    elastic_batch
    five_crop
    horizontal_flip
    pad
    perspective
    resize
    resized_crop
    rotate
    ten_crop
    vertical_flip

Meta
----------
.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    clamp_bounding_boxes
    convert_bounding_box_format
    get_dimensions
    get_num_channels
    get_num_frames
    get_size

Misc
------
.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    gaussian_blur
    gaussian_blur_batch
    normalize
    to_dtype

Temporal
--------
.. autosummary::
    :toctree: generated/
    :template: function.rst

    uniform_temporal_subsample

Type Conversion 
----------------
.. autosummary::
    :toctree: generated/
    :template: function.rst

    to_batch_images
    to_image
