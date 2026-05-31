#!/usr/bin/env python3
"""Compatibility wrapper for the renamed W-GRAM V2 PrefixLM trainer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_TARGET = Path(__file__).with_name("590_train_wgram_v2_prefixlm.py")
_SPEC = importlib.util.spec_from_file_location("wgram_v2_prefixlm_trainer_impl", _TARGET)
assert _SPEC and _SPEC.loader
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

build_arg_parser = _IMPL.build_arg_parser
train = _IMPL.train
run_generation_gate_from_checkpoint = _IMPL.run_generation_gate_from_checkpoint
main = _IMPL.main


if __name__ == "__main__":
    main()
