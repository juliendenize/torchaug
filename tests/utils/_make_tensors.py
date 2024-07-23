import torch
from torchvision.transforms.v2.functional._type_conversion import to_pil_image

from torchaug import ta_tensors
from torchaug.transforms.functional._utils._tensor import _max_value as get_max_value


BATCH_DEFAULT_SIZE = (2,)
DEFAULT_SIZE = (17, 11)


NUM_CHANNELS_MAP = {
    "GRAY": 1,
    "GRAY_ALPHA": 2,
    "RGB": 3,
    "RGBA": 4,
}


def make_image_pil(*args, **kwargs):
    return to_pil_image(make_image(*args, **kwargs))


def make_image(
    size=DEFAULT_SIZE,
    *,
    color_space="RGB",
    batch_dims=(),
    dtype=None,
    device="cpu",
    memory_format=torch.contiguous_format,
):
    num_channels = NUM_CHANNELS_MAP[color_space]
    dtype = dtype or torch.uint8
    max_value = get_max_value(dtype)
    data = torch.testing.make_tensor(
        (*batch_dims, num_channels, *size),
        low=0,
        high=max_value,
        dtype=dtype,
        device=device,
        memory_format=memory_format,
    )
    if color_space in {"GRAY_ALPHA", "RGBA"}:
        data[..., -1, :, :] = max_value

    return ta_tensors.Image(data)


def make_image_tensor(*args, **kwargs):
    return make_image(*args, **kwargs).as_subclass(torch.Tensor)


def make_batch_images(*args, batch_dims=BATCH_DEFAULT_SIZE, **kwargs):
    return ta_tensors.BatchImages(make_image_tensor(*args, batch_dims=batch_dims, **kwargs))


def make_nested_images(*args, batch_dims=BATCH_DEFAULT_SIZE, **kwargs):
    return ta_tensors.ImageNestedTensors([make_image(*args, batch_dims=[], **kwargs) for _ in range(batch_dims)])


def make_batch_images_tensor(*args, **kwargs):
    return make_batch_images(*args, **kwargs).as_subclass(torch.Tensor)


def make_bounding_boxes(
    canvas_size=DEFAULT_SIZE,
    *,
    format=ta_tensors.BoundingBoxFormat.XYXY,
    num_boxes=1,
    dtype=None,
    device="cpu",
):
    def sample_position(values, max_value):
        # We cannot use torch.randint directly here, because it only allows integer scalars as values for low and high.
        # However, if we have batch_dims, we need tensors as limits.
        return torch.stack([torch.randint(max_value - v, ()) for v in values.tolist()])

    if isinstance(format, str):
        format = ta_tensors.BoundingBoxFormat[format]

    dtype = dtype or torch.float32

    h, w = [torch.randint(1, s, (num_boxes,)) for s in canvas_size]
    y = sample_position(h, canvas_size[0])
    x = sample_position(w, canvas_size[1])

    if format is ta_tensors.BoundingBoxFormat.XYWH:
        parts = (x, y, w, h)
    elif format is ta_tensors.BoundingBoxFormat.XYXY:
        x1, y1 = x, y
        x2 = x1 + w
        y2 = y1 + h
        parts = (x1, y1, x2, y2)
    elif format is ta_tensors.BoundingBoxFormat.CXCYWH:
        cx = x + w / 2
        cy = y + h / 2
        parts = (cx, cy, w, h)
    else:
        raise ValueError(f"Format {format} is not supported")

    return ta_tensors.BoundingBoxes(
        torch.stack(parts, dim=-1).to(dtype=dtype, device=device),
        format=format,
        canvas_size=canvas_size,
    )


def make_batch_bounding_boxes(
    canvas_size=DEFAULT_SIZE,
    *,
    format=ta_tensors.BoundingBoxFormat.XYXY,
    num_boxes=1,
    batch_dims=(BATCH_DEFAULT_SIZE[0],),
    dtype=None,
    device="cpu",
):
    bboxes = []
    for _ in range(batch_dims[0]):
        bboxes.append(
            make_bounding_boxes(
                canvas_size=canvas_size,
                format=format,
                num_boxes=num_boxes,
                dtype=dtype,
                device=device,
            ).as_subclass(torch.Tensor)
        )
    bboxes = torch.cat(bboxes)
    samples_ranges = [(num_boxes * idx, num_boxes * (idx + 1)) for idx in range(batch_dims[0])]
    return ta_tensors.BatchBoundingBoxes(bboxes, format=format, canvas_size=canvas_size, samples_ranges=samples_ranges)


def make_nested_bounding_boxes(
    canvas_size=DEFAULT_SIZE,
    *,
    format=ta_tensors.BoundingBoxFormat.XYXY,
    num_boxes=1,
    batch_dims=BATCH_DEFAULT_SIZE[0],
    dtype=None,
    device="cpu",
):
    return ta_tensors.BoundingBoxesNestedTensors(
        [
            make_bounding_boxes(
                canvas_size=canvas_size,
                format=format,
                num_boxes=num_boxes,
                dtype=dtype,
                device=device,
            )
            for _ in range(batch_dims)
        ]
    )


def make_detection_masks(size=DEFAULT_SIZE, *, num_masks=1, dtype=None, device="cpu"):
    """Make a "detection" mask, i.e. (N, H, W), where each object is encoded as one of N boolean masks."""
    return ta_tensors.Mask(
        torch.testing.make_tensor(
            (num_masks, *size),
            low=0,
            high=2,
            dtype=dtype or torch.bool,
            device=device,
        )
    )


def make_batch_detection_masks(
    size=DEFAULT_SIZE,
    *,
    num_masks=1,
    batch_dims=BATCH_DEFAULT_SIZE,
    dtype=None,
    device="cpu",
):
    """Make a batch of "detection" masks, i.e. (N*B, H, W), where each object is encoded as one of N boolean
    masks.
    """
    masks = []
    for _ in range(batch_dims[0]):
        masks.append(
            torch.testing.make_tensor(
                (num_masks, *size),
                low=0,
                high=2,
                dtype=dtype or torch.bool,
                device=device,
            )
        )
    masks = torch.cat(masks)
    samples_ranges = [(num_masks * idx, num_masks * (idx + 1)) for idx in range(batch_dims[0])]
    return ta_tensors.BatchMasks(masks, samples_ranges=samples_ranges)


def make_nested_detection_masks(
    size=DEFAULT_SIZE,
    *,
    num_masks=1,
    batch_dims=BATCH_DEFAULT_SIZE,
    dtype=None,
    device="cpu",
):
    return ta_tensors.MaskNestedTensors(
        [
            make_detection_masks(
                size=size,
                num_masks=num_masks,
                dtype=dtype,
                device=device,
            )
            for _ in range(batch_dims)
        ]
    )


def make_segmentation_mask(size=DEFAULT_SIZE, *, num_categories=10, batch_dims=(), dtype=None, device="cpu"):
    """Make a "segmentation" mask, i.e. (*, H, W), where the category is encoded as pixel value."""
    return ta_tensors.Mask(
        torch.testing.make_tensor(
            (*batch_dims, *size),
            low=0,
            high=num_categories,
            dtype=dtype or torch.uint8,
            device=device,
        )
    )


def make_batch_segmentation_masks(
    size=DEFAULT_SIZE,
    *,
    num_categories=10,
    batch_dims=BATCH_DEFAULT_SIZE,
    dtype=None,
    device="cpu",
):
    """Make a batch of "segmentation" masks, i.e. (B, *, H, W), where the category is encoded as pixel value."""
    masks = []
    for _ in range(batch_dims[0]):
        masks.append(
            torch.testing.make_tensor(
                (*batch_dims, *size),
                low=0,
                high=2,
                dtype=dtype or torch.uint8,
                device=device,
            )
        )
    masks = torch.cat(masks)
    samples_ranges = [(batch_dims[0] * idx, batch_dims[0] * (idx + 1)) for idx in range(batch_dims[0])]
    return ta_tensors.BatchMasks(masks, samples_ranges=samples_ranges)


def make_nested_segmentation_masks(
    size=DEFAULT_SIZE,
    *,
    num_categories=10,
    batch_dims=BATCH_DEFAULT_SIZE,
    dtype=None,
    device="cpu",
):
    return ta_tensors.MaskNestedTensors(
        [
            make_segmentation_mask(
                size=size,
                num_categories=num_categories,
                dtype=dtype,
                device=device,
            )
            for _ in range(batch_dims)
        ]
    )


def make_video(size=DEFAULT_SIZE, *, num_frames=3, batch_dims=(), **kwargs):
    return ta_tensors.Video(make_image(size, batch_dims=(*batch_dims, num_frames), **kwargs))


def make_video_tensor(*args, **kwargs):
    return make_video(*args, **kwargs).as_subclass(torch.Tensor)


def make_batch_videos(*args, batch_dims=BATCH_DEFAULT_SIZE, **kwargs):
    return ta_tensors.BatchVideos(make_video_tensor(*args, batch_dims=batch_dims, **kwargs))


def make_batch_videos_tensor(*args, **kwargs):
    return make_batch_videos(*args, **kwargs).as_subclass(torch.Tensor)


def make_nested_videos(*args, batch_dims=BATCH_DEFAULT_SIZE, **kwargs):
    return ta_tensors.VideoNestedTensors([make_video(*args, batch_dims=[], **kwargs) for _ in range(batch_dims)])


def make_labels(size=DEFAULT_SIZE, *, batch_dims=(), **kwargs):
    return ta_tensors.Labels(torch.randint(0, 10, (*batch_dims, *size), **kwargs))


def make_batch_labels(size=DEFAULT_SIZE, *, batch_dims=BATCH_DEFAULT_SIZE, **kwargs):
    samples_ranges = [(idx * size[0], (idx + 1) * size[0]) for idx in range(batch_dims[0])]
    return ta_tensors.BatchLabels(
        torch.randint(0, 10, (batch_dims[0] * size[0], *size[1:]), **kwargs), samples_ranges=samples_ranges
    )


def make_nested_labels(*args, batch_dims=BATCH_DEFAULT_SIZE, **kwargs):
    return ta_tensors.LabelsNestedTensors([make_labels(*args, batch_dims=[], **kwargs) for _ in range(batch_dims)])


SAMPLE_MAKERS = [
    make_image,
    make_bounding_boxes,
    make_segmentation_mask,
    make_detection_masks,
    make_video,
]
BATCH_MAKERS = [
    make_batch_images,
    make_batch_bounding_boxes,
    make_batch_segmentation_masks,
    make_batch_detection_masks,
    make_batch_videos,
]

IMAGE_MAKERS = [make_image, make_batch_images]
IMAGE_TENSOR_AND_MAKERS = SAMPLE_MAKERS + [make_image_tensor]
IMAGE_AND_VIDEO_TENSOR_AND_MAKERS = IMAGE_TENSOR_AND_MAKERS + [make_video_tensor]

BATCH_IMAGES_TENSOR_AND_MAKERS = BATCH_MAKERS + [make_batch_images_tensor]
BATCH_IMAGES_AND_VIDEO_TENSOR_AND_SAMPLE_MAKERS = BATCH_IMAGES_TENSOR_AND_MAKERS + [make_batch_videos_tensor]


ALL_IMAGES_MAKERS = IMAGE_TENSOR_AND_MAKERS + BATCH_IMAGES_TENSOR_AND_MAKERS
ALL_MAKERS = IMAGE_AND_VIDEO_TENSOR_AND_MAKERS + BATCH_IMAGES_AND_VIDEO_TENSOR_AND_SAMPLE_MAKERS

BOUNDING_BOXES_MAKERS = [make_bounding_boxes, make_batch_bounding_boxes]
VIDEO_MAKERS = [make_video, make_batch_videos]

SAMPLE_MASK_MAKERS = [make_segmentation_mask, make_detection_masks]
BATCH_MASK_MAKERS = [make_batch_segmentation_masks, make_batch_detection_masks]
MASKS_MAKERS = SAMPLE_MASK_MAKERS + BATCH_MASK_MAKERS

NESTED_MAKERS = [
    make_nested_bounding_boxes,
    make_nested_detection_masks,
    make_nested_images,
    make_nested_labels,
    make_nested_segmentation_masks,
    make_nested_videos,
]
