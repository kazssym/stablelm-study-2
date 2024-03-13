#!/usr/bin/env python3

"""test_export.py
"""

from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForCausalLM
from onnxruntime import SessionOptions, GraphOptimizationLevel

main_export(
    "stabilityai/stablelm-3b-4e1t",
    output="./exported/intermediate",
    trust_remote_code=True,
    # dtype="fp16",
    # device="dml",
)

session_options = SessionOptions()
session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
model = ORTModelForCausalLM.from_pretrained(
    "./exported/intermediate",
    provider="DmlExecutionProvider",
    session_options=session_options,
)

model.save_pretrained("./exported")
