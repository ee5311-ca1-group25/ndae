"""Data module for NDAE: exemplar loading, timeline, and sampling."""

from .exemplar import ExemplarDataset
from .sampling import (
    CropSampleSpec,
    apply_crop_spec,
    apply_take_spec,
    random_crop,
    random_take,
    sample_frame_indices,
    sample_random_crop_spec,
    sample_random_take_spec,
    stratified_uniform,
)
from .timeline import Timeline

__all__ = [
    "ExemplarDataset",
    "Timeline",
    "CropSampleSpec",
    "random_crop",
    "random_take",
    "sample_random_crop_spec",
    "sample_random_take_spec",
    "apply_crop_spec",
    "apply_take_spec",
    "stratified_uniform",
    "sample_frame_indices",
]
