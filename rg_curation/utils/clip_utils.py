import os
import sys
import pandas as pd
import numpy as np
import torch
import glob
import logging
import gc
import psutil
from PIL import Image

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def truncate_text(text, max_words=120):
    """Truncate text to a maximum number of words."""
    if not text or pd.isna(text):
        return "No impressions available"

    words = str(text).split()
    if len(words) <= max_words:
        return text
    
    return ' '.join(words[:max_words]) + '...'

def extract_all_slices(ct_volume):
    """
    Extract ALL 2D slices from 3D CT volume for comprehensive CLIP processing.
    Returns all axial, sagittal, and coronal slices.
    """
    # Ensure we have the right shape (remove channel dimension if present)
    if ct_volume.ndim == 4:
        ct_volume = ct_volume[0]  # Remove channel dimension
    
    h, w, d = ct_volume.shape
    
    # Extract all slices in each orientation
    axial_slices = []     # Looking down from top (z-axis)
    sagittal_slices = []  # Side view (x-axis) 
    coronal_slices = []   # Front view (y-axis)
    
    # Axial slices (iterate through depth dimension)
    for z in range(d):
        axial_slices.append(ct_volume[:, :, z])
    
    # Sagittal slices (iterate through height dimension)
    for x in range(h):
        sagittal_slices.append(ct_volume[x, :, :])
    
    # Coronal slices (iterate through width dimension) 
    for y in range(w):
        coronal_slices.append(ct_volume[:, y, :])
    
    return axial_slices, sagittal_slices, coronal_slices

def convert_to_rgb_image(slice_2d):
    """
    Convert a 2D CT slice to RGB PIL Image for CLIP processing.
    """
    # Normalize to 0-1 range
    slice_norm = np.clip(slice_2d, 0, 1)
    
    # Convert to 0-255 uint8
    slice_uint8 = (slice_norm * 255).astype(np.uint8)
    
    # Convert to RGB (3 channels)
    slice_rgb = np.stack([slice_uint8, slice_uint8, slice_uint8], axis=-1)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(slice_rgb, 'RGB')
    
    return pil_image