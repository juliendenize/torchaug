import re

import pytest
from torchvision import datasets

from torchaug.data.dataset import wrap_dataset_for_transforms_v2


class TestDatasetWrapper:
    def test_unknown_type(self):
        unknown_object = object()
        with pytest.raises(
            TypeError, match=re.escape("is meant for subclasses of `torchvision.datasets.VisionDataset`")
        ):
            wrap_dataset_for_transforms_v2(unknown_object)

    def test_unknown_dataset(self):
        class MyVisionDataset(datasets.VisionDataset):
            pass

        dataset = MyVisionDataset("root")

        with pytest.raises(TypeError, match="No wrapper exist"):
            wrap_dataset_for_transforms_v2(dataset)

    def test_missing_wrapper(self):
        dataset = datasets.FakeData()

        with pytest.raises(TypeError, match="please open an issue"):
            wrap_dataset_for_transforms_v2(dataset)

    def test_subclass(self, mocker):
        from torchaug.data.dataset import _dataset_wrapper

        sentinel = object()
        mocker.patch.dict(
            _dataset_wrapper.WRAPPER_FACTORIES,
            clear=False,
            values={datasets.FakeData: lambda dataset, target_keys: lambda idx, sample: sentinel},
        )

        class MyFakeData(datasets.FakeData):
            pass

        dataset = MyFakeData()
        wrapped_dataset = wrap_dataset_for_transforms_v2(dataset)

        assert wrapped_dataset[0] is sentinel
