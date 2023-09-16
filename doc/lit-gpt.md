# lit-gpt

## Collect a full model

Unquantized, falcon 40b

```bash
ansible-playbook model/falcon-7b/install.yml
```

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
python3 -m pip install -r requirements.txt
```

## Convert to lit-gpt format

> checkpoints directories are rigid, lit-gpt parses the names of the directory
> to know what type of model it is using.  We'll just use a symlink to our already 
> downloaded model while preserving lit-gpt's naming structure

```bash
mkdir -p checkpoints/tiiuae
ln -s ~/llm/model/falcon-40b-instruct/model checkpoints/tiiuae/falcon-40b-instruct
```

```bash
python scripts/download.py \
    --repo_id tiiuae/falcon-7b

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

```bash
python finetune/lora.py \
  --checkpoint_dir "checkpoints/tiiuae/falcon-7b" \
  --data_dir "data/alpaca" \
  --out_dir "out/lora/alpaca"
```
