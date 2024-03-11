#!/usr/bin/env python3

"""test_export.py
"""

from optimum.exporters.onnx import main_export
from torch import float16

main_export(
    "stabilityai/stablelm-3b-4e1t",
    output="./exported",
    trust_remote_code=True,
    # torch_dtype=float16,
    # device="dml",
)
