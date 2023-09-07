# prompt

LLM's use specific prompting mechanisms.
If the provided prompt is not formatted correctly the LLM will generate poor resulting tokens.

Many different models use similar prompt mechanisms/styles:

- Instruct  
- Orca (prompt style)
- Vicuna (prompt style)
- Chat

Each model *should* provide this information in the model card.

### Layout

This directory contains example and specific prompts for each listed type of prompting mechanism/style.  Many models reuse the same prompt styles, therefore it was more appropriate to store each type in this prompt directory rather than within each model directory.

```
prompt/
├── input
├── instruct
└── orca
```

### `input`

Sample data that is used for model analysis input.
Used for testing and comparing model's abilies with similar inputs.

### `instruct`

These are sample prompts that are used to demonstrate the instruct style of prompting

### `orca`

The orca model uses a specific prompt style, different from other models. Use the prompts as an example of effective "orca-style" prompt writing.
