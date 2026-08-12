"""Fail when a SentencePiece tokenizer produces unknowns on LM corpus files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    totals = {'lines': 0, 'pieces': 0, 'unknown_pieces': 0}
    examples: list[dict[str, object]] = []
    for path in args.inputs:
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.rstrip('\n')
                if not text:
                    continue
                ids = tokenizer.encode(text, out_type=int)
                unknowns = ids.count(tokenizer.unk_id())
                totals['lines'] += 1
                totals['pieces'] += len(ids)
                totals['unknown_pieces'] += unknowns
                if unknowns and len(examples) < 10:
                    examples.append({
                        'path': str(path),
                        'line': line_number,
                        'unknown_pieces': unknowns,
                        'text': text,
                    })

    result = {
        'tokenizer': str(args.tokenizer),
        'vocab_size': tokenizer.get_piece_size(),
        **totals,
        'unknown_fraction': (
            totals['unknown_pieces'] / totals['pieces'] if totals['pieces'] else 0
        ),
        'examples': examples,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if totals['unknown_pieces'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

