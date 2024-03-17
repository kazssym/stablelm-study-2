#!/usr/bin/env python3

"""test_run_optimum.py
"""

from onnxruntime import SessionOptions, GraphOptimizationLevel
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import torch_directml

session_options = SessionOptions()
session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.optimized_model_filepath = "./optimized.onnx"

device = torch_directml.device()

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
model = ORTModelForCausalLM.from_pretrained(
    "kazssym/stablelm-3b-4e1t-onnx",
    provider="DmlExecutionProvider",
    session_options=session_options,
).to(device)

inputs = tokenizer("The weather is always wonderful",
                   return_tensors="pt").to(model.device)
tokens = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.75,
    top_p=0.95,
    do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
