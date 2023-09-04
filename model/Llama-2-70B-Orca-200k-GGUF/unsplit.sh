#!/bin/bash
cat llama-2-70b-orca-200k.Q8_0.gguf-split-{a,b} > llama-2-70b-orca-200k.Q8_0.gguf
rm llama-2-70b-orca-200k.Q8_0.gguf-split-a
rm llama-2-70b-orca-200k.Q8_0.gguf-split-b
