from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CenterSpatialCropd,
    NormalizeIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandAxisFlipd,
    Lambdad
)

import torch


class LoadPTd:
    """Load PyTorch tensor files (.pt) from disk."""
    
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                # Load the .pt file
                d[key] = torch.load(d[key], map_location='cpu')
        return d

transforms_image_seg = Compose(
    [
        LoadImaged(keys=["image", "seg"]),
        EnsureChannelFirstd(keys=["image", "seg"]),
        Orientationd(keys=["image", "seg"], axcodes="RAS"),
        Spacingd(
            keys=["image", "seg"], pixdim=(1.5, 1.5, 3), mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image", "seg"], spatial_size=[224, 224, 160]),
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image", "seg"],
        ),
        ToTensord(keys=["image", "seg"]),
    ]
)

transforms_image = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image"],
        ),
        ToTensord(keys=["image"]),
    ]
)

transforms_embedding = Compose(
    [
        LoadPTd(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
)

transforms_embedding_tensor = Compose(
    [
        Lambdad(keys=["image"], func=lambda x: torch.tensor(x, dtype=torch.float32)),
        ToTensord(keys=["image"]),
    ]
)

# transforms_image = Compose(
#     [
#         LoadImaged(keys=["image"]),
#         EnsureChannelFirstd(keys=["image"]),
#         Orientationd(keys=["image"], axcodes="RAS"),
#         Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
#         # ScaleIntensityRanged(
#         #     keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
#         # ),
#         SpatialPadd(keys=["image"], spatial_size=[224, 224, 160], mode='constant', constant_values=-3000),
#         CenterSpatialCropd(
#             roi_size=[224, 224, 160],
#             keys=["image"],
#         ),
#         ToTensord(keys=["image"]),
#     ]
# )

transforms_image_mr = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.3645, 0.3645, 1.4), mode=("bilinear")),
        NormalizeIntensityd(keys=["image"]),
        SpatialPadd(keys=["image"], spatial_size=[384, 384, 80]),
        CenterSpatialCropd(
            roi_size=[384, 384, 80],
            keys=["image"],
        ),
        ToTensord(keys=["image"]),
    ]
)

transforms_image_verse = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image"],
        ),
        ToTensord(keys=["image"]),
    ]
)

transforms_image_augment = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image"],
        ),
        # RandRotated(keys=["image"], prob=0.16, range_x=(-15, 15), mode='bilinear', padding_mode="zeros"),
        RandZoomd(keys=["image"], prob=0.16, min_zoom=0.8, max_zoom=1.2, mode='bilinear', padding_mode="constant", value=0),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.5), invert_image=False, retain_stats=True),
        # RandAxisFlipd(keys=["image"], prob=0.5),
        ToTensord(keys=["image"]),
    ]
)

transforms_embedding_nibabel = Compose(
    [
        LoadImaged(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
)




