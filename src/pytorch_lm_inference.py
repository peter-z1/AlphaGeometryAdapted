"""PyTorch checkpoint inference adapter for AlphaGeometry proof search."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch

from train_demo_lm import CausalTransformerLM


class PytorchLanguageModelInference:
    """Expose a trained demo LM checkpoint through AlphaGeometry's beam API."""

    def __init__(
        self,
        checkpoint: str | Path,
        tokenizer: str | Path,
        batch_size: int = 4,
        max_decode_len: int = 32,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.max_decode_len = max_decode_len
        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = CausalTransformerLM(**payload["model_config"])
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def _encode_prompt(self, inputs: str) -> list[int]:
        ids = [self.bos_id]
        ids.extend(self.sp.encode(inputs, out_type=int))
        return ids

    def _continuation_text(self, generated: list[int]) -> str:
        return self.sp.decode([i for i in generated if i not in {self.bos_id, self.eos_id, self.pad_id}])

    @torch.no_grad()
    def beam_decode(self, inputs: str, eos_tokens: list[str]):
        if eos_tokens != [";"]:
            raise ValueError("only ';' is supported as an EOS token")

        prefix = self._encode_prompt(inputs)
        beams: list[tuple[float, list[int], bool]] = [(0.0, [], False)]

        for _ in range(self.max_decode_len):
            candidates: list[tuple[float, list[int], bool]] = []
            for score, generated, done in beams:
                if done:
                    candidates.append((score, generated, done))
                    continue

                context = (prefix + generated)[-self.model.block_size :]
                input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
                logits = self.model(input_ids)[0, -1]
                logits[self.pad_id] = -float("inf")
                logits[self.bos_id] = -float("inf")
                probs = torch.log_softmax(logits, dim=-1)
                values, indices = torch.topk(probs, k=min(self.batch_size, probs.numel()))

                for value, index in zip(values.tolist(), indices.tolist()):
                    new_generated = generated + [int(index)]
                    text = self._continuation_text(new_generated).strip()
                    finished = index == self.eos_id or text.endswith(";")
                    candidates.append((score + float(value), new_generated, finished))

            candidates.sort(key=lambda item: item[0], reverse=True)
            beams = candidates[: self.batch_size]
            if beams and all(done for _, _, done in beams):
                break

        seqs = []
        scores = []
        for score, generated, _done in beams:
            text = self._continuation_text(generated).strip()
            if ";" in text:
                text = text[: text.index(";") + 1]
            seqs.append(text)
            scores.append(score)

        return {"seqs_str": seqs, "scores": scores}
