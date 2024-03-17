# Description

This is a repository for testing scripts to export StableLM ONNX models from [stabilityai/stablelm-3b-4e1t](https://huggingface.co/stabilityai/stablelm-3b-4e1t).

They might require a [modified version](https://github.com/kazssym/optimum/tree/feature/stablelm) of Hugging Face Optimum.

  - test_export.py

    This script exports a StableLM model into ONNX with optional float16 conversion.
