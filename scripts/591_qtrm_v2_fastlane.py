#!/usr/bin/env python3
"""Compatibility wrapper for the renamed W-GRAM V2 fastlane."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_TARGET = Path(__file__).with_name("591_wgram_v2_fastlane.py")
_SPEC = importlib.util.spec_from_file_location("wgram_v2_fastlane_impl", _TARGET)
assert _SPEC and _SPEC.loader
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

build_arg_parser = _IMPL.build_arg_parser
build_fastlane_plan = _IMPL.build_fastlane_plan
assert_not_duplicate = _IMPL.assert_not_duplicate
main = _IMPL.main


if __name__ == "__main__":
    main()
