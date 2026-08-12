"""Sample text from a demo AlphaGeometry causal LM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from train_demo_lm import CausalTransformerLM


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate from a demo AlphaGeometry LM.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args(argv)


@torch.no_grad()
def sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, num_samples=1)].item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = CausalTransformerLM(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()
    ids = [bos_id]
    ids.extend(sp.encode(args.prompt, out_type=int))

    for _ in range(args.max_new_tokens):
        context = ids[-model.block_size :]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)[0, -1]
        logits[pad_id] = -float("inf")
        logits[bos_id] = -float("inf")
        next_id = sample_next(logits, args.temperature, args.top_k)
        if next_id == eos_id:
            break
        ids.append(next_id)

    pieces = [token for token in ids if token not in {bos_id, eos_id, pad_id}]
    print(sp.decode(pieces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
