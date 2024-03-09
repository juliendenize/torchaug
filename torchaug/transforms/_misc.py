from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.transforms.v2._utils import _setup_number_or_seq, _setup_size

from torchaug import ta_tensors

from . import functional as F
from ._transform import RandomApplyTransform, Transform
from ._utils import (
    _parse_labels_getter,
    get_sample_or_batch_bounding_boxes,
    is_pure_tensor,
)


# TODO: do we want/need to expose this?
class Identity(Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt


class Lambda(Transform):
    """Apply a user-defined function as a transform.

    This transform does not support torchscript.

    Args:
        lambd: Lambda/function to be used for transform.
    """

    _transformed_types = (object,)

    def __init__(self, lambd: Callable[[Any], Any], *types: Type):
        super().__init__()
        self.lambd = lambd
        self.types = types or self._transformed_types

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, self.types):
            return self.lambd(inpt)
        else:
            return inpt

    def forward_single(self, flat_inputs: List[Any]) -> List[Any]:
        if self.p == 1.0:
            pass
        elif self.p == 0.0 or torch.rand(1) >= self.p:
            return flat_inputs

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform],
            num_chunks=1,
            chunks_indices=(
                torch.tensor(
                    [0],
                    device=flat_inputs[0].device if isinstance(flat_inputs[0], torch.Tensor) else "cpu",
                ),
            ),
        )[0]

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return flat_outputs

    def extra_repr(self) -> str:  # type: ignore[override]
        extras = []
        name = getattr(self.lambd, "__name__", None)
        if name:
            extras.append(name)
        extras.append(f"types={[type.__name__ for type in self.types]}")
        return ", ".join(extras)


class LinearTransformation(Transform):
    """Transform a tensor image or video with a square transformation matrix and a mean_vector computed offline.

    Given transformation_matrix and mean_vector, will flatten the `torch.Tensor` and
    subtract mean_vector from it which is then followed by computing the dot
    product with the transformation matrix and then reshaping the tensor to its
    original shape.

    Applications:
        whitening transformation: Suppose X is a column vector zero-centered data.
        Then compute the data covariance matrix [D x D] or the batch covariance matrix [B x D x D]
        with torch.mm(X.t(), X), perform SVD on this matrix and pass it as transformation_matrix.

    Args:
        transformation_matrix: tensor [D x D] or [B x D X D], D = C x H x W
        mean_vector: tensor [D] or [B X D], D = C x H x W
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    _transformed_types = (
        is_pure_tensor,
        ta_tensors.Image,
        ta_tensors.Video,
        ta_tensors.BatchImages,
        ta_tensors.BatchVideos,
    )

    def __init__(
        self,
        transformation_matrix: torch.Tensor,
        mean_vector: torch.Tensor,
    ):
        super().__init__()
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError(
                "transformation_matrix should be square. Got "
                f"{tuple(transformation_matrix.size())} rectangular matrix."
            )
        if mean_vector.size(0) != transformation_matrix.size(0):
            raise ValueError(
                f"mean_vector should have the same length {mean_vector.size(0)}"
                f" as any one of the dimensions of the transformation_matrix [{tuple(transformation_matrix.size())}]"
            )

        if transformation_matrix.device != mean_vector.device:
            raise ValueError(
                f"Input tensors should be on the same device. Got {transformation_matrix.device} and "
                f"{mean_vector.device}"
            )

        if transformation_matrix.dtype != mean_vector.dtype:
            raise ValueError(
                f"Input tensors should have the same dtype. Got {transformation_matrix.dtype} and {mean_vector.dtype}"
            )

        self.transformation_matrix = transformation_matrix
        self.mean_vector = mean_vector

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        shape = inpt.shape
        n = shape[-3] * shape[-2] * shape[-1]

        if n != self.transformation_matrix.shape[0]:
            raise ValueError(
                "Input tensor and transformation matrix have incompatible shape."
                + f"[{shape[-3]} x {shape[-2]} x {shape[-1]}] != "
                + f"{self.transformation_matrix.shape[0]}"
            )

        if inpt.device != self.mean_vector.device:
            raise ValueError(
                "Input tensor should be on the same device as transformation matrix and mean vector. "
                f"Got {inpt.device} vs {self.mean_vector.device}"
            )

        flat_inpt = inpt.reshape(-1, n) - self.mean_vector
        transformation_matrix = self.transformation_matrix.to(flat_inpt.dtype)

        output = torch.mm(flat_inpt, transformation_matrix)
        output = output.reshape(shape)

        if isinstance(
            inpt,
            (
                ta_tensors.Image,
                ta_tensors.Video,
                ta_tensors.BatchImages,
                ta_tensors.BatchVideos,
            ),
        ):
            output = ta_tensors.wrap(output, like=inpt)
        return output


class Normalize(Transform):
    """Normalize a tensor image or video with mean and standard deviation.

    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
        inplace: whether to apply the transform in-place.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        inplace: bool = False,
    ):
        super().__init__()
        self.mean = list(mean)
        self.std = list(std)
        self.inplace = inplace

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.normalize, inpt, mean=self.mean, std=self.std, inplace=self.inplace)


class RandomGaussianBlur(RandomApplyTransform):
    """Blurs image with randomly chosen Gaussian blur.

    The input is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
        p: probability of applying the transform.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        kernel_size: Union[int, Sequence[int]],
        sigma: Union[int, float, Sequence[float]] = (0.1, 2.0),
        p: float = 0.5,
        batch_inplace: bool = False,
        batch_transform: bool = False,
    ) -> None:
        super().__init__(p=p, batch_inplace=batch_inplace, batch_transform=batch_transform)
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        self.sigma = _setup_number_or_seq(sigma, "sigma")

        if not 0.0 < self.sigma[0] <= self.sigma[1]:
            raise ValueError(f"sigma values should be positive and of the form (min, max). Got {self.sigma}")

    def _get_params(
        self,
        flat_inputs: List[Any],
        num_chunks: int,
        chunks_indices: Tuple[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        params = []

        for _ in range(num_chunks):
            if not self.batch_transform:
                sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
                params.append({"sigma": [sigma, sigma]})
            else:
                device = chunks_indices[0].device
                batch_size = chunks_indices[0].shape[0]
                sigma = (
                    torch.empty((batch_size, 1), device=device)
                    .uniform_(self.sigma[0], self.sigma[1])
                    .expand(batch_size, 2)
                )
            params.append({"sigma": sigma})

        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.gaussian_blur_batch if self.batch_transform else F.gaussian_blur,  # type: ignore[arg-type]
            inpt,
            self.kernel_size,
            **params,
        )


class GaussianBlur(RandomGaussianBlur):
    """Blurs image with randomly chosen Gaussian blur.

    If the input is a Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
        batch_inplace: whether to apply the batch transform in-place.
            Does not prevent functionals to make copy but can reduce time and memory consumption.
        batch_transform: whether to apply the transform in batch mode.
    """

    def __init__(
        self,
        kernel_size: Union[int, Sequence[int]],
        sigma: Union[int, float, Sequence[float]] = (0.1, 2.0),
        batch_inplace: bool = False,
        batch_transform: bool = False,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            sigma=sigma,
            p=1,
            batch_inplace=batch_inplace,
            batch_transform=batch_transform,
        )


class ToDtype(Transform):
    """Converts the input to a specific dtype, optionally scaling the values for images or videos.

    .. note::
        ``ToDtype(dtype, scale=True)`` is the recommended replacement for ``ConvertImageDtype(dtype)``.

    Args:
        dtype: The dtype to convert to.
            If a ``torch.dtype`` is passed, e.g. ``torch.float32``, only images and videos will be converted
            to that dtype: this is for compatibility with :class:`torchvision.transforms.v2.ConvertImageDtype`.
            A dict can be passed to specify per-ta_tensor conversions, e.g.
            ``dtype={ta_tensors.Image: torch.float32, ta_tensors.Mask: torch.int64, "others":None}``. The "others"
            key can be used as a catch-all for any other ta_tensor type, and ``None`` means no conversion.
        scale: Whether to scale the values for images or videos.
    """

    def __init__(
        self,
        dtype: Union[torch.dtype, Dict[Union[Type, str], Optional[torch.dtype]]],
        scale: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(dtype, (dict, torch.dtype)):
            raise ValueError(f"dtype must be a dict or a torch.dtype, got {type(dtype)} instead")

        if (
            isinstance(dtype, dict)
            and torch.Tensor in dtype
            and any(
                cls in dtype
                for cls in [
                    ta_tensors.Image,
                    ta_tensors.Video,
                    ta_tensors.BatchImages,
                    ta_tensors.BatchVideos,
                ]
            )
        ):
            warnings.warn(
                "Got `dtype` values for `torch.Tensor` and either `ta_tensors.Image` or `ta_tensors.Video`. "
                "Note that a plain `torch.Tensor` will *not* be transformed by this (or any other transformation) "
                "in case a `ta_tensors.Image` or `ta_tensors.Video` is present in the input."
            )
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(self.dtype, torch.dtype):
            # For consistency / BC with ConvertImageDtype, we only care about images or videos when dtype
            # is a simple torch.dtype
            if not is_pure_tensor(inpt) and not isinstance(
                inpt,
                (
                    ta_tensors.Image,
                    ta_tensors.Video,
                    ta_tensors.BatchImages,
                    ta_tensors.BatchVideos,
                ),
            ):
                return inpt

            dtype: Optional[torch.dtype] = self.dtype
        elif type(inpt) in self.dtype:
            dtype = self.dtype[type(inpt)]
        elif "others" in self.dtype:
            dtype = self.dtype["others"]
        else:
            raise ValueError(
                f"No dtype was specified for type {type(inpt)}. "
                "If you only need to convert the dtype of images or videos, you can just pass "
                "e.g. dtype=torch.float32. "
                "If you're passing a dict as dtype, "
                'you can use "others" as a catch-all key '
                'e.g. dtype={ta_tensors.Mask: torch.int64, "others": None} to pass-through the rest of the inputs.'
            )

        supports_scaling = is_pure_tensor(inpt) or isinstance(
            inpt,
            (
                ta_tensors.Image,
                ta_tensors.BatchImages,
                ta_tensors.Video,
                ta_tensors.BatchVideos,
            ),
        )
        if dtype is None:
            if self.scale and supports_scaling:
                warnings.warn(
                    "scale was set to True but no dtype was specified for images or videos: no scaling will be done."
                )
            return inpt

        return self._call_kernel(F.to_dtype, inpt, dtype=dtype, scale=self.scale)


class SanitizeBoundingBoxes(Transform):
    """Remove degenerate/invalid bounding boxes and their corresponding labels and masks.

    This transform removes bounding boxes and their associated labels/masks that:

    - are below a given ``min_size``: by default this also removes degenerate boxes that have e.g. X2 <= X1.
    - have any coordinate outside of their corresponding image. You may want to
      call :class:`torchvision.transforms.v2.ClampBoundingBoxes` first to avoid undesired removals.

    It is recommended to call it at the end of a pipeline, before passing the
    input to the models. It is critical to call this transform if
    :class:`torchvision.transforms.v2.RandomIoUCrop` was called.
    If you want to be extra careful, you may call it after all transforms that
    may modify bounding boxes but once at the end should be enough in most
    cases.

    Args:
        min_size The size below which bounding boxes are removed.
        labels_getter: indicates how to identify the labels in the input.
            By default, this will try to find a "labels" key in the input (case-insensitive), if
            the input is a dict or it is a tuple whose second element is a dict.
            This heuristic should work well with a lot of datasets, including the built-in torchvision datasets.
            It can also be a callable that takes the same input
            as the transform, and returns the labels.
    """

    def __init__(
        self,
        min_size: float = 1.0,
        labels_getter: Optional[Union[Callable[[Any], Optional[torch.Tensor]], str]] = "default",
    ) -> None:
        super().__init__()

        if min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {min_size}.")
        self.min_size = min_size

        self.labels_getter = labels_getter
        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs: Any) -> Any:
        inputs = inputs if len(inputs) > 1 else inputs[0]

        labels = self._labels_getter(inputs)
        if labels is not None and not isinstance(labels, torch.Tensor):
            raise ValueError(
                f"The labels in the input to forward() must be a tensor or None, got {type(labels)} instead."
            )

        flat_inputs, spec = tree_flatten(inputs)
        boxes = get_sample_or_batch_bounding_boxes(flat_inputs)

        if labels is not None and boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of boxes (shape={boxes.shape}) and number of labels (shape={labels.shape}) do not match."
            )

        is_batch_boxes = isinstance(boxes, ta_tensors.BatchBoundingBoxes)

        if not is_batch_boxes:
            boxes = cast(
                ta_tensors.BoundingBoxes,
                F.convert_bounding_box_format(
                    boxes,
                    new_format=ta_tensors.BoundingBoxFormat.XYXY,
                ),
            )
        else:
            boxes = cast(
                ta_tensors.BatchBoundingBoxes,
                F.convert_bounding_box_format(
                    boxes,
                    new_format=ta_tensors.BoundingBoxFormat.XYXY,
                ),
            )

        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        valid = (ws >= self.min_size) & (hs >= self.min_size) & (boxes >= 0).all(dim=-1)  # type: ignore[attr-defined]
        # TODO: Do we really need to check for out of bounds here? All
        # transforms should be clamping anyway, so this should never happen?
        image_h, image_w = boxes.canvas_size
        valid &= (boxes[:, 0] <= image_w) & (boxes[:, 2] <= image_w)
        valid &= (boxes[:, 1] <= image_h) & (boxes[:, 3] <= image_h)

        params = {"valid": valid.as_subclass(torch.Tensor), "labels": labels}
        flat_outputs = [
            # Even-though it may look like we're transforming all inputs, we don't:
            # _transform() will only care about BoundingBoxeses and the labels
            self._transform(inpt, params)
            for inpt in flat_inputs
        ]

        return tree_unflatten(flat_outputs, spec)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_label = inpt is not None and inpt is params["labels"]
        is_bounding_boxes = isinstance(inpt, (ta_tensors.BoundingBoxes, ta_tensors.BatchBoundingBoxes))
        is_mask = isinstance(inpt, (ta_tensors.Mask, ta_tensors.BatchMasks))
        is_bounding_boxes_or_mask = is_bounding_boxes or is_mask

        if not (is_label or is_bounding_boxes_or_mask):
            return inpt

        if is_bounding_boxes or isinstance(inpt, ta_tensors.BatchMasks):
            output = inpt.masked_remove(inpt, mask=~params["valid"])
            return output
        else:
            output = inpt[params["valid"]]

        if is_label:
            return output

        return ta_tensors.wrap(output, like=inpt)
