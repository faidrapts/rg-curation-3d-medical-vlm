"""
CT scan preprocessing utilities.

Provides slice extraction with center-of-mass cropping for 3D CT volumes, and
a helper to obtain the project's standard MONAI transform pipeline
(``transforms_image`` from ``rg_curation.utils.monai_transforms``).

After ``transforms_image`` is applied, voxel intensities are already scaled to
``[0, 1]``.  All downstream slice conversion functions assume this range.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)


def optimal_center_mass_crop(
    slice_2d: np.ndarray,
    crop_ratio: float = 0.85,
    fallback_to_original: bool = True,
) -> np.ndarray:
    """Center-of-mass crop for a 2D CT slice.

    Uses a conservative crop ratio to remove scanner table background while
    preserving anatomical content.

    Args:
        slice_2d: 2D CT slice as a NumPy array (Hounsfield units or
            pre-normalized values).
        crop_ratio: Fraction of the original image dimensions to retain
            (default: 0.85).
        fallback_to_original: If ``True``, return the original slice whenever
            cropping would produce an invalid result.

    Returns:
        Cropped 2D slice as a NumPy array.
    """
    try:
        from skimage.morphology import binary_closing, disk, remove_small_objects

        # Tissue detection: use HU threshold for raw CT, percentile for normalized.
        if slice_2d.min() < -500:
            binary = slice_2d > -200
        else:
            normed = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            binary = normed > np.percentile(normed, 12)

        binary = remove_small_objects(binary, min_size=100)
        binary = binary_closing(binary, disk(3))

        try:
            center_y, center_x = ndimage.center_of_mass(binary)
            center_y, center_x = int(center_y), int(center_x)
        except Exception:
            center_y = slice_2d.shape[0] // 2
            center_x = slice_2d.shape[1] // 2

        crop_h = max(int(slice_2d.shape[0] * crop_ratio), slice_2d.shape[0] // 4)
        crop_w = max(int(slice_2d.shape[1] * crop_ratio), slice_2d.shape[1] // 4)

        y_min = max(0, center_y - crop_h // 2)
        y_max = min(slice_2d.shape[0], center_y + crop_h // 2)
        x_min = max(0, center_x - crop_w // 2)
        x_max = min(slice_2d.shape[1], center_x + crop_w // 2)

        # Adjust window when it runs past image edges.
        if y_max - y_min < crop_h:
            if y_max == slice_2d.shape[0]:
                y_min = max(0, y_max - crop_h)
            else:
                y_max = min(slice_2d.shape[0], y_min + crop_h)
        if x_max - x_min < crop_w:
            if x_max == slice_2d.shape[1]:
                x_min = max(0, x_max - crop_w)
            else:
                x_max = min(slice_2d.shape[1], x_min + crop_w)

        cropped = slice_2d[y_min:y_max, x_min:x_max]

        area_ratio = (cropped.shape[0] * cropped.shape[1]) / (
            slice_2d.shape[0] * slice_2d.shape[1]
        )
        if area_ratio > 0.95 or area_ratio < 0.10:
            return slice_2d if fallback_to_original else cropped

        return cropped

    except Exception:
        return slice_2d if fallback_to_original else slice_2d


def extract_center_slices(
    volume: torch.Tensor,
    num_slices: int = 10,
    slice_spacing: int = 2,
    crop_ratio: float = 0.85,
) -> Tuple[np.ndarray, List[int]]:
    """Extract evenly-spaced center slices from a 3D CT volume.

    Slices are spaced by ``slice_spacing`` around the axial center of the
    volume and each is center-of-mass cropped before being returned.

    Args:
        volume: 3D tensor of shape ``(C, H, W, D)`` or ``(H, W, D)``.
        num_slices: Number of slices to extract (default: 10).
        slice_spacing: Spacing between consecutive extracted slices
            (default: 2).
        crop_ratio: Crop ratio forwarded to
            :func:`optimal_center_mass_crop` (default: 0.85).

    Returns:
        Tuple of:
            - ``cropped_slices``: NumPy array of shape
              ``(num_unique_slices, H', W')``.
            - ``slice_indices``: Corresponding depth indices into the volume.
    """
    if volume.dim() == 4:
        volume = volume.squeeze(0)

    depth = volume.shape[-1]
    center_idx = depth // 2
    start_offset = (num_slices // 2) * slice_spacing

    raw_indices = [
        max(0, min(depth - 1, center_idx - start_offset + i * slice_spacing))
        for i in range(num_slices)
    ]
    slice_indices = sorted(set(raw_indices))

    cropped_slices = [
        optimal_center_mass_crop(volume[:, :, idx].numpy(), crop_ratio=crop_ratio)
        for idx in slice_indices
    ]
    return np.array(cropped_slices), slice_indices


def convert_slice_to_rgb(slice_2d: np.ndarray) -> Image.Image:
    """Convert a pre-normalised 2D CT slice to an RGB PIL image.

    Assumes intensities are already in ``[0, 1]`` (as produced by
    ``transforms_image`` from ``monai_transforms.py``).  Values are clipped to
    this range, then scaled to uint8 and replicated across three channels.

    Args:
        slice_2d: 2D NumPy array with values in ``[0, 1]``.

    Returns:
        PIL ``Image`` in mode ``"RGB"``.
    """
    normed = np.clip(slice_2d, 0, 1)
    uint8 = (normed * 255).astype(np.uint8)
    rgb = np.stack([uint8, uint8, uint8], axis=-1)
    return Image.fromarray(rgb, "RGB")


def get_ct_transforms():
    """Return the project's standard CT preprocessing pipeline.

    Imports and returns ``transforms_image`` from
    :mod:`rg_curation.utils.monai_transforms`.  The pipeline:

    - Loads ``.nii.gz`` files via MONAI
    - Reorients to RAS
    - Resamples to 1.5 × 1.5 × 3 mm voxels
    - Scales intensities from ``[-1000, 1000]`` HU to ``[0, 1]``
    - Pads / center-crops to ``[224, 224, 160]``

    Returns:
        A :class:`monai.transforms.Compose` pipeline that accepts a dict
        with an ``"image"`` key (path to ``.nii.gz``) and returns the same
        dict with the preprocessed tensor.
    """
    from rg_curation.utils.monai_transforms import transforms_image
    return transforms_image
