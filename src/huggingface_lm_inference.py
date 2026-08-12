"""Hugging Face/PEFT inference adapter for AlphaGeometry proof search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class HuggingFaceLanguageModelInference:
    """Expose a causal Hugging Face model through AlphaGeometry's beam API.

    Prompts are passed directly to the base causal model.  In particular, no
    chat template, natural-language instruction, or thinking tokens are added.
    This matches the raw prompt-completion objective used by the Qwen3.5
    experiment.
    """

    def __init__(
        self,
        model_name: str,
        revision: str = "",
        adapter: str | Path = "",
        batch_size: int = 4,
        max_decode_len: int = 32,
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        cache_dir: str | Path = "",
    ):
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "install experiments/qwen3_5_9b/requirements.txt to use the "
                "Hugging Face backend"
            ) from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("the Hugging Face backend requested CUDA, but no GPU is visible")
        if load_in_4bit and device != "cuda":
            raise ValueError("four-bit inference requires --device cuda")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype: {dtype}")

        self.batch_size = batch_size
        self.max_decode_len = max_decode_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision or None,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("model tokenizer has neither a pad nor an end token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "cache_dir": str(cache_dir) if cache_dir else None,
            "revision": revision or None,
            "dtype": dtype_map[dtype],
            "low_cpu_mem_usage": True,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype_map[dtype],
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": 0}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required when --adapter is supplied") from exc
            self.model = PeftModel.from_pretrained(self.model, str(adapter))

        self.device = torch.device("cuda:0" if device == "cuda" else "cpu")
        if not load_in_4bit:
            self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _truncate_at_semicolon(text: str) -> str:
        text = text.strip()
        if ";" in text:
            return text[: text.index(";") + 1]
        return text

    @torch.no_grad()
    def beam_decode(self, inputs: str, eos_tokens: list[str]):
        if eos_tokens != [";"]:
            raise ValueError("only ';' is supported as an EOS string")

        # Training encodes the prompt with this separator before the target.
        prompt = inputs.rstrip() + " "
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_decode_len,
            "num_beams": self.batch_size,
            "num_return_sequences": self.batch_size,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "stop_strings": [";"],
            "tokenizer": self.tokenizer,
        }
        if self.batch_size > 1:
            generation_kwargs["early_stopping"] = True

        try:
            generated = self.model.generate(**encoded, **generation_kwargs)
        except TypeError as exc:
            # Older compatible generation backends may not expose stop_strings;
            # max_new_tokens plus deterministic post-truncation remains safe.
            if "stop_strings" not in str(exc) and "tokenizer" not in str(exc):
                raise
            generation_kwargs.pop("stop_strings", None)
            generation_kwargs.pop("tokenizer", None)
            generated = self.model.generate(**encoded, **generation_kwargs)

        sequences = generated.sequences[:, input_length:]
        texts = [
            self._truncate_at_semicolon(
                self.tokenizer.decode(sequence, skip_special_tokens=True)
            )
            for sequence in sequences
        ]
        sequence_scores = getattr(generated, "sequences_scores", None)
        if sequence_scores is None:
            scores = [0.0] * len(texts)
        else:
            scores = [float(score) for score in sequence_scores.detach().cpu().tolist()]
        return {"seqs_str": texts, "scores": scores}
