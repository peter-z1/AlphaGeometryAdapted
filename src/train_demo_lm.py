"""Train a small causal LM on AlphaGeometry-style text.

This is a pragmatic demonstration trainer, not the paper-faithful Meliad path.
It uses the same text/tokenizer artifacts we prepare for AlphaGeometry-style
pretraining and produces a compact PyTorch checkpoint that can be sampled with
``src/generate_demo_lm.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import sentencepiece as spm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class LineDataset(Dataset[str]):
    """A seekable text-line dataset that keeps only byte offsets in memory."""

    def __init__(self, path: Path, max_lines: int = 0):
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
                    if max_lines and len(self.offsets) >= max_lines:
                        break

        if not self.offsets:
            raise ValueError(f"{path} contains no non-empty lines")

    def __len__(self) -> int:
        return len(self.offsets)

    def _handle(self):
        if self._file is None:
            self._file = self.path.open("rb")
        return self._file

    def __getitem__(self, index: int) -> str:
        handle = self._handle()
        handle.seek(self.offsets[index])
        return handle.readline().decode("utf-8").rstrip("\n")


class SentencePieceCollator:
    def __init__(self, tokenizer_path: Path, block_size: int):
        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.block_size = block_size
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def __call__(self, lines: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        full_len = self.block_size + 1

        for line in lines:
            ids = [self.bos_id]
            ids.extend(self.sp.encode(line, out_type=int))
            ids.append(self.eos_id)
            ids = ids[:full_len]
            if len(ids) < full_len:
                ids.extend([self.pad_id] * (full_len - len(ids)))

            input_ids = ids[:-1]
            labels = ids[1:]
            labels = [token if token != self.pad_id else -100 for token in labels]
            input_rows.append(input_ids)
            label_rows.append(labels)

        return torch.tensor(input_rows, dtype=torch.long), torch.tensor(label_rows, dtype=torch.long)


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.block_size = block_size
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.block_size}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = input_ids.eq(self.pad_id)
        x = self.blocks(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return self.lm_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def infinite_batches(loader: DataLoader) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        losses.append(float(loss.item()))
        if batch_index >= max_batches:
            break
    model.train()
    return sum(losses) / max(1, len(losses))


def learning_rate(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(
    out_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    model_config: dict[str, object],
    metrics: dict[str, float],
) -> None:
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "model_config": model_config,
        "metrics": metrics,
    }
    step_path = out_dir / f"checkpoint_step_{step}.pt"
    latest_path = out_dir / "checkpoint_latest.pt"
    torch.save(payload, step_path)
    torch.save(payload, latest_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small AlphaGeometry demo LM.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/demo_lm/combined_small"))
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--max_val_batches", type=int, default=50)
    parser.add_argument("--max_train_lines", type=int, default=0)
    parser.add_argument("--max_val_lines", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume_from", type=Path)
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="Load model weights from --resume_from but reset optimizer and step count.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    collator = SentencePieceCollator(args.tokenizer, args.block_size)
    train_ds = LineDataset(args.train, max_lines=args.max_train_lines)
    val_ds = LineDataset(args.val, max_lines=args.max_val_lines)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=False,
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_config = {
        "vocab_size": collator.vocab_size,
        "block_size": args.block_size,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "pad_id": collator.pad_id,
    }
    model = CausalTransformerLM(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if args.resume_weights_only:
            start_step = 0
            resume_event = "resumed_weights_only"
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_step = int(checkpoint["step"])
            resume_event = "resumed"
        print(
            json.dumps(
                {
                    "event": resume_event,
                    "checkpoint": str(args.resume_from),
                    "step": start_step,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    config_path = args.out_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "model_config": model_config,
                "num_parameters": count_parameters(model),
                "train_examples": len(train_ds),
                "val_examples": len(val_ds),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_path = args.out_dir / "metrics.jsonl"
    start_time = time.time()
    batches = infinite_batches(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "vocab_size": collator.vocab_size,
                "num_parameters": count_parameters(model),
                "train_examples": len(train_ds),
                "val_examples": len(val_ds),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    last_metrics: dict[str, float] = {}

    for step in range(start_step + 1, args.max_steps + 1):
        step_loss = 0.0
        step_tokens = 0
        lr = learning_rate(step, args.max_steps, args.warmup_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(args.grad_accum_steps):
            input_ids, labels = next(batches)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            step_tokens += int(labels.ne(-100).sum().item())

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            step_loss += float(loss.item())

        if args.clip_grad_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            metrics = {
                "step": float(step),
                "train_loss": step_loss,
                "lr": lr,
                "tokens": float(step_tokens),
                "tokens_per_sec": step_tokens / max(1e-9, elapsed),
                "elapsed_sec": elapsed,
            }
            last_metrics = metrics
            with metrics_path.open("a", encoding="utf-8") as dst:
                dst.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps(metrics, sort_keys=True), flush=True)

        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss = evaluate(model, val_loader, device, args.max_val_batches)
            metrics = {
                "step": float(step),
                "val_loss": val_loss,
                "val_ppl": math.exp(min(20.0, val_loss)),
                "elapsed_sec": time.time() - start_time,
            }
            last_metrics.update(metrics)
            with metrics_path.open("a", encoding="utf-8") as dst:
                dst.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps(metrics, sort_keys=True), flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args.out_dir, step, model, optimizer, args, model_config, last_metrics)

    print(json.dumps({"event": "done", "out_dir": str(args.out_dir)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
