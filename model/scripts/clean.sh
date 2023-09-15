#!/bin/bash
find . -type f \
	    -name "*.bin" \
	-or -name "*.gguf" \
	-or -name "*.gguf-split*" \
| xargs -I {} -P0 rm -rf {}
