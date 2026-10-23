"""Embedding backends. Clips and text queries land in the same vector space.

TorchEmbedder runs the full model on the GPU (Apple Metal or NVIDIA CUDA) when present.
OnnxEmbedder runs quantized ONNX weights with onnxruntime (CPU, or CUDA via onnxruntime-gpu).
"""

import json
import os
import warnings

import numpy as np

from .models import ModelSize


def _quiet():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore")
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def load_embedder(size: ModelSize):
    _quiet()
    return TorchEmbedder(size) if size.backend == "torch" else OnnxEmbedder(size)


class TorchEmbedder:
    def __init__(self, size: ModelSize):
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        # bf16 halves memory and is faster on GPU; fp16 is unsafe for this model (silent NaNs).
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.device = device
        self.model = SentenceTransformer(str(size.dir), device=device, model_kwargs={"torch_dtype": dtype})

    def embed_clips(self, clips: list[np.ndarray], batch_size: int = 4) -> np.ndarray:
        """clips: list of (n, H, W, 3) uint8 frame stacks -> (len, dim) unit vectors."""
        vecs = self.model.encode([{"video": c} for c in clips], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, prompt_name="query", normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)


class OnnxEmbedder:
    """Two ONNX graphs: vision_encoder turns frame patches into features; model takes
    token ids plus those features and returns the pooled sentence embedding."""

    def __init__(self, size: ModelSize):
        import onnxruntime as ort
        from transformers import AutoProcessor

        available = ort.get_available_providers()
        # CoreML is deliberately skipped: compiling this model needs many GB of RAM.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available \
            else ["CPUExecutionProvider"]
        self.device = "cuda" if providers[0].startswith("CUDA") else "cpu"
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        onnx_dir = size.dir / "onnx"
        self.vision = ort.InferenceSession(str(onnx_dir / f"vision_encoder{size.onnx_suffix}.onnx"), opts,
                                           providers=providers)
        self.model = ort.InferenceSession(str(onnx_dir / f"model{size.onnx_suffix}.onnx"), opts,
                                          providers=providers)
        self.processor = AutoProcessor.from_pretrained(str(size.dir))
        prompts = json.loads((size.dir / "config_sentence_transformers.json").read_text())["prompts"]
        self.query_prompt = prompts["query"]
        self.empty = np.zeros((0, 512), np.float32)

    def _run(self, inputs, video_features=None) -> np.ndarray:
        feeds = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "image_features": self.empty,
            "video_features": self.empty if video_features is None else video_features,
            "audio_features": self.empty,
        }
        return self.model.run(["sentence_embedding"], feeds)[0][0]

    def embed_clips(self, clips: list[np.ndarray], batch_size: int = 4) -> np.ndarray:
        out = []
        for frames in clips:
            x = self.processor(text=["<|video|>"], videos=[frames], return_tensors="np")
            feats = self.vision.run(None, {"pixel_values": x["pixel_values_videos"].astype(np.float32),
                                           "pixel_position_ids": x["video_position_ids"]})[0]
            out.append(self._run(x, feats))
        return _normalize(np.stack(out))

    def embed_query(self, text: str) -> np.ndarray:
        x = self.processor(text=[self.query_prompt + text], return_tensors="np")
        return _normalize(self._run(x))
