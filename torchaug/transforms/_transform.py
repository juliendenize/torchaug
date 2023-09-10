from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class RandomTransform(nn.Module, ABC):
    """Abstract class to make a base class for all random transforms.

    args:
        p: Probability to apply the transform.
    """

    def __init__(self, p: float) -> None:
        super().__init__()

        if not 0 <= p <= 1:
            raise ValueError(
                f"p should be superior to 0 (included) and inferior to 1 (included). Got {p}."
            )

        self.p = p

    @abstractmethod
    def apply_transform(self, img: Tensor):
        """Function to perform transformation on the image.

        .. note::
            Should be overridden by subclasses.

        Args:
            img: Image to transform.

        Return:
            The transformed image.
        """
        ...

    def forward(self, img: Tensor):
        """Call :meth:`apply_transform` based on random sample.

        Args:
            img: Image to randomly transform based on :attr:`~p`.

        Return:
            The randomly transformed image.
        """
        if self.p == 0.0:
            return img
        elif self.p == 1.0 or torch.rand(1).item() < self.p:
            return self.apply_transform(img)
        return img


class VideoBase(ABC):
    """Abstract class to make a base class for all video transforms.

    Args:
        video_format: Dimension order of the video. Can be ``TCHW`` or ``CTHW``.
    """

    def __init__(self, video_format: str) -> None:
        super().__init__()
        self.check_format(video_format)
        self._video_format = video_format

    @staticmethod
    def check_format(format: str) -> None:
        """Check if the format is either ``TCHW`` or ``CTHW``. Raises an error if not.

        Args:
            format: Format to check.
        """
        if format not in ["CTHW", "TCHW"]:
            raise ValueError(
                f"video_format should be either 'CTHW' or 'TCHW'. Got {format}."
            )

    @property
    def video_format(self):
        """Dimension order of the video.

        Can be ``TCHW`` or ``CTHW``.
        """
        return self._video_format

    @video_format.setter
    def video_format(self, format: str) -> None:
        self.check_format(format)
        self._video_format = format

    @property
    def time_before_channel(self) -> bool:
        """Boolean that checks if the :attr:`~video_format` has time dimension before channel."""
        if self.video_format == "CTHW":
            return False
        elif self.video_format == "TCHW":
            return True
        else:
            raise ValueError("Attribute _video_format was wrongly changed by user.")
