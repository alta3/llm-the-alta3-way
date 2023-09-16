
## Checkpoint Directories

Checkpoint directory names within lit-gpt are rigid and required.
lit-gpt parses the names of the directory and uses it to know the model.
In order to preserve our model directories and not need to re-download,
create a symlink at the location that lit-gpt expects.

Example preserves llm-the-alta3-way's falcon-40b-instruct directory

```bash
mkdir -p checkpoints/tiiuae
ln -s ~/llm/model/falcon-40b-instruct/model checkpoints/tiiuae/falcon-40b-instruct
```
