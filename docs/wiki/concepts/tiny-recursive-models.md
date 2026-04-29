# Tiny Recursive Models

Tiny Recursive Models are the current reference for the QTRM recursive reasoning
core axis.

Core contract:

- maintain `z_H` and `z_L` carry states
- reset carry for halted sequences
- run most H/L cycles under `torch.no_grad()`
- run the final cycle with gradients
- detach new carry
- use a Q-head for halt/continue behavior

QTRM implication:

- `src/qtrm_mm/core.py` has z_H/z_L names and H/L loops, but it does not yet
  implement TRM carry semantics or ACT halting.
- Treat current recursive core as experimental until those gaps are closed or
  deliberately documented.
