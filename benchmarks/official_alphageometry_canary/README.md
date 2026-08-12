# Official AlphaGeometry auxiliary-construction canaries

These files evaluate the official DeepMind JAX/Meliad checkpoint rather than
the PyTorch reconstruction or a converted ChatLLM model. Official code,
Meliad, tokenizer, and checkpoint files are downloaded separately under
`external/google_deepmind_alphageometry/`; none are redistributed here.

`run_target.py` performs instrumented search while independent DD+AR candidate
checks run in CPU worker processes. For example:

```bash
.venv-official/bin/python \
  benchmarks/official_alphageometry_canary/run_target.py \
  --problem_name translated_imo_2015_p3 \
  --workers 8 \
  --model_batch_size 32 --beam_size 512 --search_depth 16
```

Before a long benchmark, use the simple `orthocenter` problem in the official
checkout to check that the model adds a nonempty construction and DD+AR then
proves the goal. Saved historical hard-target outputs are included here for
comparison. The runner is expensive and assumes a correctly installed
official JAX/Meliad environment.
