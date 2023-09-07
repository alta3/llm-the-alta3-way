#!/usr/bin/env python3
import sys
from ctransformers import AutoModelForCausalLM

model_bin = sys.argv[1]
llm = AutoModelForCausalLM.from_pretrained(model_bin, model_type="llama")
print("# v02: streaming example")

prompt = "AI is going to"
print(prompt, end=" ")
for text in llm(prompt, stream=True):
        print(text, end="", flush=True)
