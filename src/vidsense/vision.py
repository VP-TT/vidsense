"""CLIP ViT-B/32: frame embeddings, text embeddings and zero-shot visual tags.

CLIP is a dual encoder (a ViT for images, a Transformer for text) trained so that an
image and its caption land close together in one shared 512-d space. VidSense uses
that space only while processing a video: to pick keyframes, to label them zero-shot,
and to score how well each transcript segment matches each keyframe for DTW. None of
these 512-d vectors are stored in ChromaDB; retrieval happens in the MiniLM text space.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

_load_lock = threading.RLock()

# Averaging a few prompt templates is the standard CLIP zero-shot trick; it smooths
# out the wording of any single prompt.
TEMPLATES = ("a photo of {}.", "a video frame showing {}.")


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class ClipEncoder:
    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device or pick_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        self.image_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.dim = int(self.model.config.projection_dim)
        self.logit_scale = float(self.model.logit_scale.exp().item())  # ~100 for OpenAI CLIP

    def _pooled(self, output):
        # transformers 5 returns a model output whose pooler_output is the projected embedding;
        # older versions return the tensor directly.
        return output if isinstance(output, self._torch.Tensor) else output.pooler_output

    def encode_images(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        """L2-normalised (n, 512) float32 image embeddings."""
        batches = []
        for i in range(0, len(images), batch_size):
            pixels = self.image_processor(images=images[i : i + batch_size], return_tensors="pt")["pixel_values"]
            with self._torch.inference_mode():
                features = self._pooled(self.model.get_image_features(pixel_values=pixels.to(self.device)))
            batches.append(features.float().cpu().numpy())
        if not batches:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.concatenate(batches)).astype(np.float32)

    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """L2-normalised (n, 512) float32 text embeddings. CLIP reads at most 77 tokens."""
        batches = []
        for i in range(0, len(texts), batch_size):
            tokens = self.tokenizer(
                texts[i : i + batch_size], padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)
            with self._torch.inference_mode():
                features = self._pooled(self.model.get_text_features(**tokens))
            batches.append(features.float().cpu().numpy())
        if not batches:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.concatenate(batches)).astype(np.float32)


@lru_cache(maxsize=1)
def _load_clip(model_name: str) -> ClipEncoder:
    return ClipEncoder(model_name)


def load_clip(model_name: str) -> ClipEncoder:
    with _load_lock:
        return _load_clip(model_name)


def load_vocabulary(path: str | Path | None = None) -> list[str]:
    """Labels for zero-shot tagging: one per line, '#' starts a comment."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("vidsense").joinpath("resources/visual_concepts.txt").read_text(encoding="utf-8")
    labels = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return list(dict.fromkeys(labels))


class VisualTagger:
    """Zero-shot labelling: score each frame against "a photo of {label}" for every label.

    The softmax over CLIP's scaled similarities gives a probability per label; we keep
    the top few above a threshold, so ambiguous frames get no tags rather than wrong ones.
    """

    def __init__(self, encoder: ClipEncoder, labels: list[str]):
        self.encoder = encoder
        self.labels = labels
        per_template = [encoder.encode_texts([template.format(label) for label in labels]) for template in TEMPLATES]
        self.label_embeddings = normalize(np.mean(per_template, axis=0)).astype(np.float32)

    def tag(self, image_embeddings: np.ndarray, top_k: int = 3, min_prob: float = 0.1) -> list[list[tuple[str, float]]]:
        logits = self.encoder.logit_scale * (image_embeddings @ self.label_embeddings.T)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        tags = []
        for row in probs:
            best = np.argsort(row)[::-1][:top_k]
            tags.append([(self.labels[i], round(float(row[i]), 3)) for i in best if row[i] >= min_prob])
        return tags


@lru_cache(maxsize=4)
def _load_tagger(model_name: str, vocab_path: str | None) -> VisualTagger:
    return VisualTagger(load_clip(model_name), load_vocabulary(vocab_path))


def load_tagger(model_name: str, vocab_path: str | None = None) -> VisualTagger:
    with _load_lock:
        return _load_tagger(model_name, vocab_path)
