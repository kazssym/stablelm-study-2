#!/usr/bin/env python3

"""test_export.py
"""

from onnxconverter_common import float16
from onnxruntime import SessionOptions, GraphOptimizationLevel
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForCausalLM
import onnx
import os

with_float16 = True

main_export(
    "stabilityai/stablelm-3b-4e1t",
    output="./exported-intermediate",
    trust_remote_code=True,
    # dtype="fp16",
    # device="dml",
)

if with_float16:
    # Conversion to float16 by Optimum had problems with DirectML.
    model = onnx.load_model("./exported-intermediate/model.onnx")
    model = float16.convert_float_to_float16(model, disable_shape_infer=True)

    os.remove("./exported-intermediate/model.onnx_data")
    onnx.save_model(
        model, "./exported-intermediate/model.onnx",
        save_as_external_data=True,
        location="model.onnx_data",
    )

session_options = SessionOptions()
session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
model = ORTModelForCausalLM.from_pretrained(
    "./exported-intermediate",
    provider="DmlExecutionProvider",
    session_options=session_options,
)

model.save_pretrained("./exported")
