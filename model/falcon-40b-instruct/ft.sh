#!/bin/bash
export LIT_GPT=../../git/lit-gpt
export DATA_ALPACA=../../dataset/alpaca
mkdir -p checkpoint

python3 -m pip install \
    --pre torch torchvision torchaudio numpy  \
    --index-url https://download.pytorch.org/whl/nightly/cu118

python3 -m pip install -r ${LIT_GPT}/requirements.txt \
	tokenizers \
	sentencepiece

python3 ${LIT_GPT}/scripts/convert_hf_checkpoint.py \
	--checkpoint_dir checkpoint

python3 scripts/prepare_alpaca.py \
	--destination_path ${DATA_ALPACA} \
	--checkpoint_dir checkpoint
