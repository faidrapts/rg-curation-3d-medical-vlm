"""
CLIP-score computation for scan–report pairs.

For 3D CT volumes the vision embedding is the L2-normalised mean of per-slice
CLIP embeddings (slices are extracted with the same center-crop strategy as
DreamSim).  For 2D images a single CLIP vision embedding is used.

The CLIP score is the cosine similarity between the averaged vision embedding
and the text embedding of the corresponding radiology report.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "datacomp_xl_s13b_b90k"


def setup_clip_model(
    model_name: str = _DEFAULT_MODEL,
    pretrained: str = _DEFAULT_PRETRAINED,
    compile_model: bool = True,
) -> Tuple:
    """Load an OpenCLIP model.

    Args:
        model_name: OpenCLIP architecture string (default: ``"ViT-B-32"``).
        pretrained: Pretrained weights tag (default:
            ``"datacomp_xl_s13b_b90k"``).
        compile_model: Apply ``torch.compile`` on GPU when available
            (default: ``True``).

    Returns:
        Tuple of ``(model, preprocess, tokenizer, device)``.
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if compile_model and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
            logger.info("torch.compile applied to CLIP model")
        except Exception as exc:
            logger.warning(f"torch.compile skipped: {exc}")

    model.eval()
    logger.info(f"OpenCLIP {model_name} ({pretrained}) loaded on {device}")
    return model, preprocess, tokenizer, device


def _encode_slices(
    slices: np.ndarray,
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode a stack of 2D slices with the CLIP vision encoder.

    Args:
        slices: Array of shape ``(N, H, W)``.
        model: OpenCLIP model.
        preprocess: CLIP preprocessing transform.
        device: Target device.
        batch_size: Number of slices per GPU step (default: 64).

    Returns:
        L2-normalised float tensor of shape ``(N, D)``.
    """
    from rg_curation.utils.ct_preprocessing import convert_slice_to_rgb

    features: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(slices), batch_size):
            batch = slices[start : start + batch_size]
            imgs = torch.stack(
                [preprocess(convert_slice_to_rgb(s).resize((224, 224), Image.LANCZOS))
                 for s in batch]
            ).to(device, non_blocking=True)
            f = model.encode_image(imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            features.append(f.cpu())
    return torch.cat(features, dim=0)


_CLIP_MAX_TOKENS = 77  # CLIP's hard context-window limit


def compute_clip_score_ct(
    cropped_slices: np.ndarray,
    findings_text: str,
    model,
    preprocess,
    tokenizer,
    device: torch.device,
    vision_batch_size: int = 64,
    max_tokens: int = _CLIP_MAX_TOKENS,
) -> float:
    """Compute the CLIP alignment score for one CT–report pair.

    The score is the cosine similarity between the mean slice vision
    embedding and the text embedding of the findings.

    Args:
        cropped_slices: Array of shape ``(N, H, W)`` — pre-cropped CT slices.
        findings_text: Radiology findings string.
        model: OpenCLIP model.
        preprocess: CLIP preprocessing transform.
        tokenizer: OpenCLIP tokenizer.
        device: Target device.
        vision_batch_size: Slices per GPU step (default: 64).
        max_tokens: Word-level truncation limit applied before tokenisation.
            Defaults to CLIP's 77-token context window.

    Returns:
        Cosine similarity score clipped to ``[-1, 1]``.
    """
    from rg_curation.utils.text_utils import truncate_text

    text = truncate_text(findings_text, max_tokens=max_tokens)

    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        vision_feats = _encode_slices(
            cropped_slices, model, preprocess, device, batch_size=vision_batch_size
        ).to(device)
        vision_avg = vision_feats.mean(dim=0, keepdim=True)
        vision_avg = vision_avg / vision_avg.norm(dim=-1, keepdim=True)

        score = (vision_avg @ text_feat.T).item()

    return float(np.clip(score, -1.0, 1.0))


def compute_clip_score_2d(
    image_path: str,
    findings_text: str,
    model,
    preprocess,
    tokenizer,
    device: torch.device,
    max_tokens: int = _CLIP_MAX_TOKENS,
) -> float:
    """Compute the CLIP alignment score for one 2D image–report pair.

    Args:
        image_path: Path to a PNG/JPEG image.
        findings_text: Radiology findings string.
        model: OpenCLIP model.
        preprocess: CLIP preprocessing transform.
        tokenizer: OpenCLIP tokenizer.
        device: Target device.
        max_tokens: Word-level truncation limit.  Defaults to CLIP's
            77-token context window.

    Returns:
        Cosine similarity score clipped to ``[-1, 1]``.
    """
    from rg_curation.utils.text_utils import truncate_text

    text = truncate_text(findings_text, max_tokens=max_tokens)

    img = Image.open(image_path).convert("RGB").resize((224, 224), Image.LANCZOS)

    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        img_tensor = preprocess(img).unsqueeze(0).to(device)
        vision_feat = model.encode_image(img_tensor)
        vision_feat = vision_feat / vision_feat.norm(dim=-1, keepdim=True)

        score = (vision_feat @ text_feat.T).item()

    return float(np.clip(score, -1.0, 1.0))


def compute_clip_scores_batch(
    slices_list: List[np.ndarray],
    findings_list: List[str],
    model,
    preprocess,
    tokenizer,
    device: torch.device,
    vision_batch_size: int = 64,
    max_tokens: int = _CLIP_MAX_TOKENS,
) -> List[float]:
    """Batch CLIP-score computation for multiple CT–report pairs.

    Encodes all text strings in a single forward pass and encodes vision
    slices in batches, grouping all slices from all scans together for
    maximum GPU utilisation.

    Args:
        slices_list: List of ``(N_i, H, W)`` cropped-slice arrays, one per
            scan.
        findings_list: Radiology findings strings, same length as
            ``slices_list``.
        model: OpenCLIP model.
        preprocess: CLIP preprocessing transform.
        tokenizer: OpenCLIP tokenizer.
        device: Target device.
        vision_batch_size: GPU batch size for vision encoding (default: 64).
        max_tokens: Word-level truncation limit.  Defaults to CLIP's
            77-token context window.

    Returns:
        List of per-scan CLIP scores clipped to ``[-1, 1]``.
    """
    from rg_curation.utils.ct_preprocessing import convert_slice_to_rgb
    from rg_curation.utils.text_utils import truncate_text

    n = len(slices_list)
    texts = [truncate_text(t, max_tokens=max_tokens) for t in findings_list]

    with torch.no_grad():
        text_tokens = tokenizer(texts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        all_imgs: List[torch.Tensor] = []
        scan_idx_map: List[int] = []
        for i, slices in enumerate(slices_list):
            for s in slices:
                pil = convert_slice_to_rgb(s).resize((224, 224), Image.LANCZOS)
                all_imgs.append(preprocess(pil))
                scan_idx_map.append(i)

        all_vision_feats: List[torch.Tensor] = []
        for start in range(0, len(all_imgs), vision_batch_size):
            batch = torch.stack(all_imgs[start : start + vision_batch_size]).to(
                device, non_blocking=True
            )
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            all_vision_feats.append(f)
        vision_feats = torch.cat(all_vision_feats, dim=0)

        scan_idx_arr = np.array(scan_idx_map)
        scores: List[float] = []
        for i in range(n):
            mask = torch.from_numpy(scan_idx_arr == i).to(device)
            avg = vision_feats[mask].mean(dim=0, keepdim=True)
            avg = avg / avg.norm(dim=-1, keepdim=True)
            score = (avg @ text_feats[i : i + 1].T).item()
            scores.append(float(np.clip(score, -1.0, 1.0)))

    return scores
