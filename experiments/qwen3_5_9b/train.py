"""Train a Qwen3.5-9B LoRA adapter on AlphaGeometry-style text.

Two objectives are supported:

* ``pretrain`` predicts every token in a synthetic theorem/proof string;
* ``auxiliary`` predicts only the hidden-construction completion, while prompt
  tokens receive the ignored label ``-100``.

The heavy Hugging Face dependencies are imported only for a real training run,
so ``--validate_only`` remains useful on login and CPU nodes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_TRANSFORMERS = (5, 3, 0)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"stage", "model", "data", "peft", "training", "output_dir"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration is missing: {', '.join(missing)}")
    if config["stage"] not in {"pretrain", "auxiliary"}:
        raise ValueError("stage must be 'pretrain' or 'auxiliary'")
    if not config["model"].get("name"):
        raise ValueError("model.name is required")
    if config["model"].get("quantization", "none") not in {"none", "4bit"}:
        raise ValueError("model.quantization must be 'none' or '4bit'")
    for field in ("train", "val", "max_length"):
        if field not in config["data"]:
            raise ValueError(f"data.{field} is required")
    if int(config["data"]["max_length"]) <= 0:
        raise ValueError("data.max_length must be positive")
    if config["data"].get("overflow", "error") not in {"error", "truncate_left"}:
        raise ValueError("data.overflow must be 'error' or 'truncate_left'")


def version_tuple(version: str) -> tuple[int, int, int]:
    pieces: list[int] = []
    for part in version.split(".")[:3]:
        digits = "".join(character for character in part if character.isdigit())
        pieces.append(int(digits or 0))
    return tuple((pieces + [0, 0, 0])[:3])  # type: ignore[return-value]


class OffsetDataset(Dataset[dict[str, str]]):
    """Seekable line dataset that stores offsets rather than the corpus text."""

    def __init__(self, path: Path, max_rows: int = 0):
        self.path = path
        self.offsets: list[int] = []
        self._file = None
        with path.open("rb") as src:
            while True:
                offset = src.tell()
                line = src.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
                    if max_rows and len(self.offsets) >= max_rows:
                        break
        if not self.offsets:
            raise ValueError(f"{path} contains no non-empty rows")

    def __len__(self) -> int:
        return len(self.offsets)

    def _handle(self):
        if self._file is None:
            self._file = self.path.open("rb")
        return self._file

    def raw_line(self, index: int) -> str:
        handle = self._handle()
        handle.seek(self.offsets[index])
        return handle.readline().decode("utf-8").rstrip("\n")


class SyntaxDataset(OffsetDataset):
    def __getitem__(self, index: int) -> dict[str, str]:
        return {"text": self.raw_line(index)}


class AuxiliaryDataset(OffsetDataset):
    def __getitem__(self, index: int) -> dict[str, str]:
        row = json.loads(self.raw_line(index))
        return {"prompt": str(row["prompt"]), "target": str(row["target"])}


def add_eos(ids: list[int], eos_id: int | None) -> list[int]:
    if eos_id is not None and (not ids or ids[-1] != eos_id):
        return ids + [eos_id]
    return ids


def encode_syntax(
    tokenizer: Any,
    text: str,
    max_length: int,
    overflow: str,
) -> tuple[list[int], list[int]]:
    ids = add_eos(tokenizer.encode(text, add_special_tokens=False), tokenizer.eos_token_id)
    if len(ids) > max_length:
        if overflow == "error":
            raise ValueError(
                f"syntax example has {len(ids)} tokens, exceeding max_length={max_length}; "
                "run audit_tokenizer.py or choose overflow='truncate_left'"
            )
        ids = ids[-max_length:]
    return ids, list(ids)


def encode_auxiliary(
    tokenizer: Any,
    prompt: str,
    target: str,
    max_length: int,
    overflow: str,
) -> tuple[list[int], list[int]]:
    # The trailing space matches the inference adapter and separates x00 from
    # the first construction point without adding chat or instruction tokens.
    prompt_ids = tokenizer.encode(prompt.rstrip() + " ", add_special_tokens=False)
    target_ids = add_eos(
        tokenizer.encode(target.strip(), add_special_tokens=False), tokenizer.eos_token_id
    )
    excess = len(prompt_ids) + len(target_ids) - max_length
    if excess > 0:
        if overflow == "error":
            raise ValueError(
                f"auxiliary example has {len(prompt_ids) + len(target_ids)} tokens, "
                f"exceeding max_length={max_length}; run audit_tokenizer.py"
            )
        if excess >= len(prompt_ids):
            raise ValueError("the auxiliary target alone does not fit in max_length")
        prompt_ids = prompt_ids[excess:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return input_ids, labels


class GeometryCollator:
    """Tokenize and dynamically right-pad syntax or prompt-completion rows."""

    def __init__(
        self,
        tokenizer: Any,
        stage: str,
        max_length: int,
        overflow: str,
        pad_to_multiple_of: int = 8,
    ):
        self.tokenizer = tokenizer
        self.stage = stage
        self.max_length = max_length
        self.overflow = overflow
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer must have a pad token before constructing batches")

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        encoded: list[tuple[list[int], list[int]]] = []
        for row in rows:
            if self.stage == "pretrain":
                encoded.append(
                    encode_syntax(
                        self.tokenizer,
                        row["text"],
                        self.max_length,
                        self.overflow,
                    )
                )
            else:
                encoded.append(
                    encode_auxiliary(
                        self.tokenizer,
                        row["prompt"],
                        row["target"],
                        self.max_length,
                        self.overflow,
                    )
                )

        padded_length = max(len(item[0]) for item in encoded)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            padded_length = ((padded_length + multiple - 1) // multiple) * multiple

        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for input_ids, labels in encoded:
            padding = padded_length - len(input_ids)
            input_rows.append(input_ids + [self.pad_id] * padding)
            label_rows.append(labels + [-100] * padding)
            attention_rows.append([1] * len(input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        }


def resolved_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": config["stage"],
        "model": config["model"]["name"],
        "revision": config["model"].get("revision"),
        "cache_dir": (
            str(resolve_repo_path(config["model"]["cache_dir"]))
            if config["model"].get("cache_dir")
            else None
        ),
        "quantization": config["model"].get("quantization", "none"),
        "train": str(resolve_repo_path(config["data"]["train"])),
        "train_exists": resolve_repo_path(config["data"]["train"]).exists(),
        "val": str(resolve_repo_path(config["data"]["val"])),
        "val_exists": resolve_repo_path(config["data"]["val"]).exists(),
        "adapter_from": (
            str(resolve_repo_path(config["adapter_from"]))
            if config.get("adapter_from")
            else None
        ),
        "output_dir": str(resolve_repo_path(config["output_dir"])),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="validate configuration and paths without importing Hugging Face packages",
    )
    parser.add_argument("--resume_from_checkpoint", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    summary = resolved_summary(config)
    if args.validate_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["train_exists"] and summary["val_exists"] else 2

    try:
        import transformers
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "install experiments/qwen3_5_9b/requirements.txt before training"
        ) from exc

    if version_tuple(transformers.__version__) < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"transformers>={'.'.join(map(str, MIN_TRANSFORMERS))} is required for "
            f"Qwen3.5 text-only loading; found {transformers.__version__}"
        )
    unsupported_training_keys = sorted(
        set(config["training"]) - set(inspect.signature(TrainingArguments).parameters)
    )
    if unsupported_training_keys:
        raise ValueError(
            "training options unsupported by the installed Transformers version: "
            + ", ".join(unsupported_training_keys)
        )
    if not summary["train_exists"] or not summary["val_exists"]:
        raise FileNotFoundError("prepared data is missing; run prepare_data.py first")

    seed = int(config["training"].get("seed", 35))
    set_seed(seed)
    model_config = config["model"]
    model_name = model_config["name"]
    revision = model_config.get("revision")
    cache_dir = model_config.get("cache_dir")
    if cache_dir:
        cache_dir = str(resolve_repo_path(cache_dir))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, revision=revision
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Qwen tokenizer has neither a pad nor an end token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype_name = model_config.get("dtype", "bfloat16")
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported model dtype: {dtype_name}")
    dtype = dtypes[dtype_name]
    load_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "revision": revision,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if model_config.get("attn_implementation"):
        load_kwargs["attn_implementation"] = model_config["attn_implementation"]

    quantized = model_config.get("quantization", "none") == "4bit"
    if quantized:
        if not torch.cuda.is_available():
            raise RuntimeError("four-bit QLoRA requires a CUDA GPU")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        local_rank = max(0, int(os.environ.get("LOCAL_RANK", "0")))
        load_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    gradient_checkpointing = bool(config["training"].get("gradient_checkpointing", True))
    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

    if config.get("adapter_from"):
        adapter_path = resolve_repo_path(config["adapter_from"])
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"syntax adapter {adapter_path} is missing; run syntax_pretrain.json first"
            )
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        peft_config = config["peft"]
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=int(peft_config.get("r", 32)),
                lora_alpha=int(peft_config.get("lora_alpha", 64)),
                lora_dropout=float(peft_config.get("lora_dropout", 0.05)),
                bias="none",
                target_modules=peft_config.get("target_modules", "all-linear"),
                use_rslora=bool(peft_config.get("use_rslora", True)),
            ),
        )
    model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    data_config = config["data"]
    dataset_type = SyntaxDataset if config["stage"] == "pretrain" else AuxiliaryDataset
    train_dataset = dataset_type(
        resolve_repo_path(data_config["train"]), int(data_config.get("max_train_rows", 0))
    )
    val_dataset = dataset_type(
        resolve_repo_path(data_config["val"]), int(data_config.get("max_eval_rows", 0))
    )
    collator = GeometryCollator(
        tokenizer=tokenizer,
        stage=config["stage"],
        max_length=int(data_config["max_length"]),
        overflow=data_config.get("overflow", "error"),
        pad_to_multiple_of=int(data_config.get("pad_to_multiple_of", 8)),
    )

    output_dir = resolve_repo_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training_values = dict(config["training"])
    training_values["output_dir"] = str(output_dir)
    training_values.setdefault("remove_unused_columns", False)
    training_args = TrainingArguments(**training_values)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )

    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    resolved = {
        "source_config": str(args.config.resolve()),
        "resolved": summary,
        "config": config,
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "final_adapter": str(final_adapter),
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **resolved}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
