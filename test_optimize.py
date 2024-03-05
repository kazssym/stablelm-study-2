#!/usr/bin/env python3

from optimum.onnxruntime import ORTModelForCausalLM, ORTOptimizer, OptimizationConfig

optimization_config = OptimizationConfig(99, optimize_for_gpu=False)

model = ORTModelForCausalLM.from_pretrained(
    "kazssym/stablelm-3b-4e1t-onnx-fp32",
    #provider="DmlExecutionProvider",
)
optimizer = ORTOptimizer.from_pretrained(model)

optimizer.optimize(optimization_config, "optimized/")
