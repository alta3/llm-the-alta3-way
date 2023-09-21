# Falcon 40B Instruct

Goal: Interact with the model from python

### Procedure:

```bash
python3 -m venv venv
source venv/bin/activeate
python3 -m pip install transformers torch einops accelerate
python3 girafatron.py
# helpfully downloads the model for you, ~20 mins waiting
```

# Girafatron. - The name is based falcon model on using a giraffe as a prompt character

Do inference on a hugging face transformers library using pytorch. Your output is token by token.
