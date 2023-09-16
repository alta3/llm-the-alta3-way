# lit-gpt

## Setup `lit-gpt`

```bash
cd ~/llm/git/lit-gpt
python3 -m venv venv
source venv/bin/activate
```

0. Uncomment optional and torch dependencies in requirements.txt

```bash
vim requrements.txt
```

```bash
pip install \
    --index-url https://download.pytorch.org/whl/nightly/cu118 \
    --pre 'torch>=2.1.0dev'
python3 -m pip install -r requirements.txt
```

## Collect model and convert

```bash
python3 scripts/download.py --repo_id tiiuae/falcon-7b
```

```bash
python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

## Verify conversion success

```bash
python generate/base.py \
    --prompt "LLM efficiency competitions are fun, because" \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b
```

## Download, prepare, and inspect alpaca dataset

```bash
python scripts/prepare_alpaca.py \
    --checkpoint_dir checkpoints/tiiuae/falcon-7b
less data/alpaca/alpaca_data_cleaned_archive.json
```

## Finetune

Edit `finetune/lora.py` to set the `micro_match_size` to 1

```python
micro_match_size = 1
```

```bash
python finetune/lora.py \
    --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
    --data_dir "data/alpaca" \
    --out_dir "out/lora/alpaca"
```
