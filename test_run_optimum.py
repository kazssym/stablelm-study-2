#!/usr/bin/env python3

"""test_run_optimum.py

This script runs a StableLM model with Optimum and ONNX Runtime.
"""

import sys
import torch_directml
from onnxruntime import SessionOptions, GraphOptimizationLevel
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer


def main() -> int:
    """main
    """

    session_options = SessionOptions()
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = "./optimized.onnx"

    device = torch_directml.device()

    tokenizer = AutoTokenizer.from_pretrained("kazssym/stablelm-3b-4e1t-onnx")
    model = ORTModelForCausalLM.from_pretrained(
        "kazssym/stablelm-3b-4e1t-onnx",
        provider="DmlExecutionProvider",
        session_options=session_options,
    ).to(device)

    inputs = tokenizer(
        "The weather is always wonderful",
        return_tensors="pt",
    ).to(model.device)
    tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.75,
        top_p=0.95,
        do_sample=True,
    )
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
