#!/usr/bin/env python3

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from onnxruntime import SessionOptions, GraphOptimizationLevel

session_options = SessionOptions()
session_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
model = ORTModelForCausalLM.from_pretrained(
    # "kazssym/stablelm-3b-4e1t-onnx-fp32",
    "./optimized",
    provider="DmlExecutionProvider",
    session_options=session_options,
)

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
