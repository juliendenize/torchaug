import warnings

from ._color import BatchRandomGrayscale


class BatchRandomGrayScale(BatchRandomGrayscale):
    def __init__(
        self, p: float = 0.5, num_output_channels: int = 3, inplace: bool = False
    ) -> None:
        super().__init__(p, num_output_channels, inplace)
        warnings.warn(
            (
                f"{self.__class__.__name__} has been renamed to BatchRandomGrayscale for consistency with other vision libraries, "
                f"please update your import accordingly. {self.__class__.__name__} will be deleted in 0.4."
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
