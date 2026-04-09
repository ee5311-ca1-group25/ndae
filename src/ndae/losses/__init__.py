"""Public loss API for NDAE."""

from .objectives import init_loss, local_loss, overflow_loss
from .perceptual import VGG19Features
from .swd import gram_loss, gram_matrix, slice_loss, sliced_wasserstein_loss

__all__ = [
    "VGG19Features",
    "gram_matrix",
    "gram_loss",
    "sliced_wasserstein_loss",
    "slice_loss",
    "overflow_loss",
    "init_loss",
    "local_loss",
]
