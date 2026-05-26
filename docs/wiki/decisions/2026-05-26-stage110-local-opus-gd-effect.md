# 2026-05-26 Stage110 Local OPUS/GD Effect

## Decision

Stage110 is a useful partial signal, not an accepted generalization result.

```text
Static branch:
  final/best eval loss: 0.547716
  GDsuite smoke accuracy: 0.333333
  GDsuite smoke mean margin: -0.037247
  generation prefix token accuracy: 0.269755

OPUS/GD branch:
  final/best eval loss: 0.478261
  GDsuite smoke accuracy: 0.500000
  GDsuite smoke mean margin: -0.434665
  generation prefix token accuracy: 0.251534
```

OPUS/GD helped ordinary heldout loss and the coarse GD smoke accuracy, but it
hurt GD margin and free-generation prefix accuracy. Do not promote it as
accepted generalization yet.

## Plain-Language Read

OPUS/GD is starting to choose better practice problems, but the exam it sees is
still too narrow. The student got better at the short worksheet, yet the answer
confidence is less stable and open-ended speaking did not improve.

So the right conclusion is:

```text
Data selection has a real signal.
The generalization proxy is still too thin.
Long-context Generalization Dynamics coverage must be restored before a large
DGX claim can be trusted.
```

## Why This Is Not Accepted

The current BLT local model uses `seq_len=384`.

Official GDsuite coverage by byte-level shifted length:

```text
seq_len 384:    8600 / 66164 rows = 12.998%
seq_len 512:   23841 / 66164 rows = 36.033%
seq_len 768:   32128 / 66164 rows = 48.558%
seq_len 2048:  32128 / 66164 rows = 48.558%
seq_len 8192:  50164 / 66164 rows = 75.818%
seq_len 12288: 62028 / 66164 rows = 93.749%
seq_len 16384: 66164 / 66164 rows = 100.000%
```

At `seq_len=384`, the proxy mostly sees `repetitive_answer` and
`intuitive_answer`. It misses the long-context flipped, successive, and truthy
axes that are closer to the real Generalization Dynamics target.

## Implementation Fix Made

The local A/B runner had a misleading gate:

```text
output name: gdsuite_smoke_44_report.json
actual behavior before fix: full official_gdsuite_choice_probe.jsonl with max_rows=0
```

This was corrected in `scripts/622_run_local_opus_gd_blt_ablation.sh`:

```text
GD_PROBE_JSONL defaults to the 44-row smoke probe when available.
GD_MAX_ROWS is configurable.
Actions `gates` and `summarize` were added for rerunning evaluation without
retraining.
```

## Next Accepted-Probability Move

Do not run another nearby OPUS scalar tweak as the main experiment.

The next high-probability move is context restoration:

```text
1. Keep OPUS/GD as a candidate data selector.
2. Train or evaluate a long-context BLT/one-body variant.
3. Require GDsuite coverage to include the long-context flipped/successive/truthy axes.
4. Promote only if OPUS/GD improves loss, GD accuracy, GD margin, and generation together.
```

Fast falsification gate:

```text
If long-context coverage improves but OPUS/GD still lowers loss while hurting
GD margin or generation, OPUS/GD remains a data-efficiency helper, not the
generalization solution.
```

## Stage111 Long-Context Follow-Up

Stage111 tested that falsification gate locally with `seq_len=1024`.

```text
Static branch:
  final eval loss: 2.293907
  best eval loss:  2.282274
  GDsuite smoke valid rows: 24 / 44
  GDsuite smoke accuracy: 0.500000
  GDsuite smoke mean margin: -0.016180
  generation prefix token accuracy: 0.132812

OPUS/GD branch:
  final eval loss: 2.301578
  best eval loss:  2.292519
  GDsuite smoke valid rows: 24 / 44
  GDsuite smoke accuracy: 0.458333
  GDsuite smoke mean margin: -0.010899
  generation prefix token accuracy: 0.127604
```

Decision:

```text
Stage111 is also not accepted.
```

The useful part is that long-context restoration worked: the smoke gate now
evaluates 24 valid rows instead of 12, and the OPUS proxy uses `seq_len=1024`
from the checkpoint contract. The rejected part is that OPUS/GD did not beat
the static branch on loss, GD accuracy, or free generation.

Plain-language read:

```text
Giving the student a larger desk helped it see more of the exam.
It did not yet make the student better at choosing what to study.
```

Consequence:

```text
Do not promote OPUS/GD full selection to a large DGX run yet.
The next move should preserve the long-context contract but fix the selection
signal, or return to the strongest worked-before data/curriculum path and diff
why it produced better answer behavior.
```
