#!/usr/bin/env python3

from llama_cpp import Llama
from pprint import pprint

llm = Llama(model_path="../../model/Llama-2-70B-Orca-200k/model/llama-2-70b-orca-200k.Q3_K_S.gguf", n_gqa=8, n_gpu_layers=80)

output = llm("Q: Name the planets in the solar system? A: ", max_tokens=64, stop=["Q:", "\n"], echo=True)

# the pretty print (pprint) library makes complex data objects easier to read
pprint(output)
