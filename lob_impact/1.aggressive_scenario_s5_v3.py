#!/usr/bin/env python
"""
Aggressive Scenario for v3 checkpoints (24-token encoding).

Identical to 1.aggressive_scenario_s5.py except it uses 24-token encoding
(base-100 size field, vocab_size=2112) required by v3 checkpoints trained
on the HPC cluster.

Usage:
    python -u lob_impact/1.aggressive_scenario_s5_v3.py --config <config.yaml>
"""

import os
import sys

# ── Step 1: Swap encoding BEFORE any lob.* imports ──
# This makes all downstream modules (inference, validation_helpers, etc.)
# use the 24-token Vocab & Message_Tokenizer transparently.
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(script_dir)
sys.path.insert(0, parent_folder_path)

# Import 24tok module and install it as 'lob.encoding'
import lob.encoding_24tok
sys.modules['lob.encoding'] = lob.encoding_24tok

# ── Step 2: Load and run the original scenario script ──
import importlib.util
_orig_path = os.path.join(script_dir, '1.aggressive_scenario_s5.py')
_spec = importlib.util.spec_from_file_location('aggressive_scenario_s5', _orig_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
