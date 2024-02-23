import functools

import torch

from ._make_tensors import (
    make_batch_bounding_boxes,
    make_batch_detection_masks,
    make_batch_images,
    make_batch_images_tensor,
    make_batch_segmentation_masks,
    make_batch_videos,
    make_batch_videos_tensor,
    make_bounding_boxes,
    make_detection_masks,
    make_image,
    make_image_tensor,
    make_segmentation_mask,
    make_video,
    make_video_tensor,
)

assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)

__all__ = [
    "assert_equal",
    "make_batch_videos",
    "make_batch_detection_masks",
    "make_batch_images",
    "make_batch_images_tensor",
    "make_batch_segmentation_masks",
    "make_batch_videos_tensor",
    "make_bounding_boxes",
    "make_detection_masks",
    "make_image",
    "make_image_tensor",
    "make_segmentation_mask",
    "make_video",
    "make_video_tensor",
    "make_batch_bounding_boxes",
]
