#!/bin/bash
find . -type f -or \
	-name "*.bin" \
	-name "*.gguf" \
	-name "*.gguf-split*" \
| xargs -I {} -P0 rm -rf {}
