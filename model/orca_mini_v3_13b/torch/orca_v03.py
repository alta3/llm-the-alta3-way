#!/usr/bin/env python3
import sys
from ctransformers import AutoModelForCausalLM

model_bin = sys.argv[1]
gpu_layers = int(sys.argv[2])
llm = AutoModelForCausalLM.from_pretrained(model_bin, model_type="llama", gpu_layers=gpu_layers)
print("# v03: streaming gpu example")

prompt = "AI is going to"
print(prompt, end=" ")
for text in llm(prompt, stream=True):
        print(text, end="", flush=True)
