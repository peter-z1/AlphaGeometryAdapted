# Large artifact release

`manifest-v4.json` identifies the files required to reproduce the selected
educational model and its reported evaluations. The bundle layout is:

```text
alphageometry_educational_release_v4/
  models/pretrain/
  models/finetuned/
  tokenizer/
  datasets/pretraining_rich_v2/
  datasets/auxiliary_combined_v4/
  evaluations/
```

Verify a downloaded bundle from the repository root:

```bash
python scripts/verify_release.py /path/to/alphageometry_educational_release_v4
```

The manifest deliberately covers the critical model, tokenizer, and training
data files. Evaluation summaries and proof images are supporting evidence and
can be regenerated from the same model and benchmark definitions.

The bundle should be published as a versioned release/archive rather than
committed to Git. Record its permanent download URL and checksum alongside
this manifest when publication is finalized.
