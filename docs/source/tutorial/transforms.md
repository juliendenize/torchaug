# Transforms Tutorial

## Torchaug Transforms

Transforms are  [nn.Module](#torch.nn.Module) classes that when called perform a transformation on a data. The data can be either a **single sample** or a **batch of samples**. However, in order to handle the data correctly, the transforms need to know if they are supposed to work in batch mode or not. To do this, the relevant transforms have a `batch_transform` boolean attribute to indicate whether the transforms should be batched or not.

All transforms are derived from the [RandomApplyTransform](#torchaug.transforms.RandomApplyTransform):
```python
class RandomApplyTransform(nn.Module):
    _transformed_types: Tuple[Union[Type, Callable[[Any], bool]], ...] = (torch.Tensor,)
    _reshape_transform: bool = False
    _receive_flatten_inputs: bool = False

    def __init__(
        self,
        p: float = 0.5,
        batch_inplace: bool = False,
        num_chunks: int = 1,
        permute_chunks: bool = False,
        batch_transform: bool = False,
    ):
        ...
        self._num_chunks = num_chunks
        self._receive_flatten_inputs = False

    @property
    def _reshape_transform(self) -> bool:
        # Can be overriden by children.
        return False

    @property
    def num_chunks(self) -> int:
        # Can be overriden by children.
        if (num_chunks == -1 or num_chunks > 1) and not self.batch_transform:
            raise ValueError("`num_chunks` should be 1 if `batch_transform` is False.")
        elif num_chunks < -1 or num_chunks == 0:
            raise ValueError("`num_chunks` should be greater than 0 or -1.")
        self._num_chnks = num_chunks

    def _get_params(self, flat_inputs: List[Any], num_chunks: int, chunks_indices: List[torch.Tensor]) -> List[Dict[str, Any]]:
        ...

    def _transform(self, input: torch.Tensor, params: Dict[str, Any]):
        ...
```
Here is the goal of the class attribute:
- `_transformed_types`: comes from `Torchvision` and defines what types are expected. By default, the transforms can work on all kinds of [tensors](#torch.Tensor), especially [TATensors](#torchaug.ta_tensors.TATensor).

Torchaug defines 7 supplementary instance attributes or properties in the [RandomApplyTransform](#torchaug.transforms.RandomApplyTransform) described below. If they are not relevant, they are not exposed by its children and kept at their defaut value except for `p` that is passed to `1`:
- `_reshape_transform`: If a transform reshapes the input, then `batch_inplace` is not taken into account and the probability `p` cannot be another value of 1, as all elements in the batch have to be transformed.
- `_receive_flatten_inputs`: By default, a transform can accept different types of input, `dict`, `list`, ... If `_receive_flatten_inputs` is `True` then the transform expects a list of inputs. This is especially useful for [SequentialTransform](#torchaug.transforms.SequentialTransform) which changes this flag to True for its contained transforms to avoid unnecessary structure checking by `pytree`.
- `p`: Probability of applying the transform. In batch mode, it will select a sub-batch of size `batch_size * p` to transform. Otherwise, transform with probability `p`.
- `batch_inplace`: In batch mode, if `True`, update the batch in place in place of the input memory, otherwise clone the input tensors first.
- `num_chunks`: In batch mode, some transforms require parameters sampled by the method `_get_params` that cannot be vectorized such as [RandomResizedCrop](#torchaug.transforms.RandomResizedCrop). To avoid a loop, which can be long and ineficient, the batch can be split into chunks to apply the same transform to sub-batches instead of per sample or to the whole batch. If -1, each sample in the batch will be sampled differently.
- `permute_chunks`: Whether to permute elements before chunking the batch. Depending on how you the batch has ben collated, it may be useful to permute chunks but usually it is not.
- `batch_transform`: Whether to apply the transform in batch mode.


For example the [ColorJitter](#torchaug.transforms.ColorJitter) `__init__` method has the following signature:
```python
class ColorJitter(RandomColorJitter):
    def __init__(
        self,
        brightness: Optional[Union[float, Sequence[float]]] = None,
        contrast: Optional[Union[float, Sequence[float]]] = None,
        saturation: Optional[Union[float, Sequence[float]]] = None,
        hue: Optional[Union[float, Sequence[float]]] = None,
        batch_inplace: bool = False,
        num_chunks: int = 1,
        permute_chunks: bool = False,
        batch_transform: bool = False,
    ):
        ...
```

But Normalize, which has the same result whether batched or not, has the following signature:
```python
class Normalize(Transform):
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        inplace: bool = False,
    ):
        ...
```

## How to use transforms

### Basic Usage

To use transforms you can import them from the `torchaug.transforms` module just as you would for Torchvision:
```python
from torchaug.transforms import (
    ColorJitter,
    Normalize,
    RandomResizedCrop,
)

color_transform = ColorJitter(..., batch_transform=True)
normalize_transform = ColorJitter(...)
resized_crop_transform = RandomResizedCrop(..., batch_transform=True)
```

### Containers and [SequentialTransform](#torchaug.transforms.SequentialTransform)

Containers can be used to perform sequential transforms. [Compose](#torchaug.transforms.Compose) has the same behavior as [Torchvision's](#torchvision.transforms.v2.Compose).

To make it easier to have a batch transform and avoid setting parameters for all transforms, Torchaug provides the [SequentialTransform](#torchaug.transforms.SequentialTransform) container which offers the ability to override the attributes of its children thanks to its `transforms_attributes_override` argument:
```python
class SequentialTransform(Transform):
    def __init__(
        self,
        transforms: List[RandomApplyTransform],
        transforms_attributes_override: Optional[Dict[str, Any]] = {
            "inplace": True,
            "batch_inplace": True,
            "batch_transform": True,
            "num_chunks": -1,
            "permute_chunks": False,
        },
    ):
        ...
```

Therefore the following code will succeed:
```python
from torchaug.transforms import ColorJitter, RandomResizedCrop,  SequentialTransform

color_transform = ColorJitter(num_chunks=1, batch_transform=False)
resized_crop_transform = RandomResizedCrop(224, batch_transform=True)


color_and_resized_crop_transform = SequentialTransform([color_transform, resized_crop_transform])
assert color_transform.batch_transform
assert color_transform.num_chunks == 24 # Look at ColorJitter's documentation to understand why.
assert resized_crop_transform.batch_transform
```
