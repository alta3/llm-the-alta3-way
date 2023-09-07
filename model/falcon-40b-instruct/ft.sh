#!/bin/bash
export LIT_GPT=../../git/lit-gpt/scripts
export DATA_ALPACA=../../dataset/alpaca
mkdir -p checkpoint

python3 ${LIT_GPT}/convert_hf_checkpoint.py \
	--checkpoint_dir checkpoint

python3 scripts/prepare_alpaca.py \
	--destination_path ${DATA_ALPACA} \
	--checkpoint_dir checkpoint
