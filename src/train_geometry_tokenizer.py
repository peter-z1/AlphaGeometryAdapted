"""Train the AlphaGeometry-style SentencePiece word tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from pretty import MAP_SYMBOL


BASE_SYMBOLS = [
    '{C}',
    '{S}',
    '{F1}',
    '{P}',
    '{QED}',
    'x00',
    '=>',
    '=',
    '?',
    ';',
    ':',
] + sorted(set(MAP_SYMBOL) - {'='})
REF_SYMBOLS = [f'{i:02}' for i in range(100)]
DEFAULT_SYMBOLS = (
    BASE_SYMBOLS
    + REF_SYMBOLS
    + [f'▁{symbol}' for symbol in BASE_SYMBOLS + REF_SYMBOLS]
)

DEFAULT_CHECK_TEXTS = [
    '{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c {F1} x00',
    'e : C a c e 02 C b d e 03 ;',
    '{P} cong a b a c => eqangle a b b c b c a c ; {QED}',
    'f : P a b c d 04 D a b c d 05 S a b c x y z 06 O a b c d 07 '
    'M x a b 08 X a b c 09 I o a b c 10 ^ a b c d e f g h 11 / a b c d e f g h 12 ?',
]


def iter_text(path: Path, text_field: str):
    if path.suffix == '.jsonl':
        with path.open('r', encoding='utf-8') as src:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield str(row.get(text_field) or (str(row['prompt']) + ' ' + str(row['target'])))
    else:
        with path.open('r', encoding='utf-8') as src:
            for line in src:
                if line.strip():
                    yield line.rstrip('\n')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a SentencePiece tokenizer for AlphaGeometry LM strings.'
    )
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--model_prefix', type=Path, default=Path('outputs/tokenizers/ag_word_757'))
    parser.add_argument('--text_field', default='text')
    parser.add_argument('--vocab_size', type=int, default=757)
    parser.add_argument('--character_coverage', type=float, default=1.0)
    parser.add_argument('--hard_vocab_limit', action='store_true')
    parser.add_argument('--user_defined_symbols', default=','.join(DEFAULT_SYMBOLS))
    parser.add_argument('--max_sentence_length', type=int, default=16384)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument(
        '--skip_self_check',
        action='store_true',
        help='Do not validate required AlphaGeometry symbols after training.',
    )
    return parser.parse_args(argv)


def verify_tokenizer(model_path: Path, required_symbols: list[str]) -> None:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    missing = [
        symbol
        for symbol in required_symbols
        if sp.id_to_piece(sp.piece_to_id(symbol)) != symbol
    ]
    if missing:
        raise RuntimeError(
            'tokenizer is missing required user-defined symbols: '
            + ', '.join(missing)
        )

    bad_texts = []
    for text in DEFAULT_CHECK_TEXTS:
        ids = sp.encode(text, out_type=int)
        if sp.unk_id() in ids:
            pieces = sp.encode(text, out_type=str)
            bad_texts.append((text, pieces))
    if bad_texts:
        details = '\n'.join(
            f'{text}\n  pieces={pieces}' for text, pieces in bad_texts
        )
        raise RuntimeError(
            'tokenizer maps required AlphaGeometry syntax to <unk>:\n' + details
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            'sentencepiece is not installed. Install training requirements first: '
            'pip install -r requirements-training.txt'
        ) from exc

    args.model_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.txt', delete=False) as tmp:
        corpus_path = Path(tmp.name)
        for path in args.inputs:
            for text in iter_text(path, args.text_field):
                tmp.write(text + '\n')

    try:
        user_defined_symbols = [
            symbol for symbol in args.user_defined_symbols.split(',') if symbol
        ]
        spm.SentencePieceTrainer.Train(
            input=str(corpus_path),
            model_prefix=str(args.model_prefix),
            model_type='word',
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            hard_vocab_limit=args.hard_vocab_limit,
            split_by_whitespace=True,
            user_defined_symbols=user_defined_symbols,
            max_sentence_length=args.max_sentence_length,
            num_threads=args.num_threads,
            bos_id=1,
            eos_id=2,
            unk_id=0,
            pad_id=3,
        )
    finally:
        corpus_path.unlink(missing_ok=True)

    if not args.skip_self_check:
        verify_tokenizer(args.model_prefix.with_suffix('.model'), user_defined_symbols)

    print(f'wrote {args.model_prefix}.model and {args.model_prefix}.vocab')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
