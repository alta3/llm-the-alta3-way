#!/usr/bin/env python3
import sys
from ctransformers import AutoModelForCausalLM

model_bin = sys.argv[1]
llm = AutoModelForCausalLM.from_pretrained(model_bin, model_type="llama")
print("# v01: simple example")

prompt = "AI is going to"
print(prompt, end=" ")
print(llm(prompt))
