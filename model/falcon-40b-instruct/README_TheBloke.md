---
datasets:
- tiiuae/falcon-refinedweb
language:
- en
inference: false
license: apache-2.0
---

<!-- header start -->
<div style="width: 100%;">
    <img src="https://i.imgur.com/EBdldam.jpg" alt="TheBlokeAI" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="display: flex; flex-direction: column; align-items: flex-start;">
        <p><a href="https://discord.gg/theblokeai">Chat & support: my new Discord server</a></p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <p><a href="https://www.patreon.com/TheBlokeAI">Want to contribute? TheBloke's Patreon page</a></p>
    </div>
</div>
<!-- header end -->

# Falcon 40B-Instruct GGML

These files are GGCC format model files for [Falcon 40B Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct).

These files will **not** work in llama.cpp, text-generation-webui or KoboldCpp.

GGCC is a new format created in a new fork of llama.cpp that introduced this new Falcon GGML-based support: [cmp-nc/ggllm.cpp](https://github.com/cmp-nct/ggllm.cpp).

Currently these files will also not work with code that previously supported Falcon, such as LoLLMs Web UI and ctransformers. But support should be added soon.

For GGMLv3 files compatible with those UIs, [please see the old `ggmlv3` branch](https://huggingface.co/TheBloke/falcon-40b-instruct-GGML/tree/ggmlv3).

## Repositories available

* [4-bit GPTQ model for GPU inference](https://huggingface.co/TheBloke/falcon-40b-instruct-GPTQ)
* [3-bit GPTQ model for GPU inference](https://huggingface.co/TheBloke/falcon-40b-instruct-3bit-GPTQ)
* [2, 3, 4, 5, 6, 8-bit GGCT models for CPU+GPU inference](https://huggingface.co/TheBloke/falcon-40b-instruct-GGML)
* [Unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/tiiuae/falcon-40b-instruct)
  
<!-- compatibility_ggml start -->
## Compatibility

To build cmp-nct's fork of llama.cpp with Falcon support plus CUDA acceleration, please try the following steps:

```
git clone https://github.com/cmp-nct/ggllm.cpp
cd ggllm.cpp
rm -rf build && mkdir build && cd build && cmake -DGGML_CUBLAS=1 .. && cmake --build . --config Release
```

Compiling on Windows: developer cmp-nct notes: 'I personally compile it using VScode. When compiling with CUDA support using the Microsoft compiler it's essential to select the "Community edition build tools". Otherwise CUDA won't compile.'

Once compiled you can then use `bin/falcon_main` just like you would use llama.cpp. For example:
```
bin/falcon_main -t 8 -ngl 100 -b 1 -m falcon7b-instruct.ggmlv3.q4_0.bin -p "What is a falcon?\n### Response:"
```

You can specify `-ngl 100` regardles of your VRAM, as it will automatically detect how much VRAM is available to be used. 

Adjust `-t 8` (the number of CPU cores to use) according to what performs best on your system. Do not exceed the number of physical CPU cores you have.

`-b 1` reduces batch size to 1. This slightly lowers prompt evaluation time, but frees up VRAM to load more of the model on to your GPU.  If you find prompt evaluation too slow and have enough spare VRAM, you can remove this parameter.

Please see https://github.com/cmp-nct/ggllm.cpp for further details and instructions.

<!-- compatibility_ggml end -->

## Provided files
| Name | Quant method | Bits | Size | Max RAM required | Use case |
| ---- | ---- | ---- | ---- | ---- | ----- |
| falcon40b-instruct.ggccv1.q2_K.bin | q2_K | 2 | 13.74 GB | 16.24 GB | New k-quant method. Uses GGML_TYPE_Q4_K for the attention.vw and feed_forward.w2 tensors, GGML_TYPE_Q2_K for the other tensors. |
| falcon40b-instruct.ggccv1.q3_K.bin | q3_K_S | 3 | 17.98 GB | 20.48 GB | New k-quant method. Uses GGML_TYPE_Q3_K for all tensors |
| falcon40b-instruct.ggccv1.q4_K.bin | q4_K_S | 4 | 23.54 GB | 26.04 GB | New k-quant method. Uses GGML_TYPE_Q4_K for all tensors |
| falcon40b-instruct.ggccv1.q5_K.bin | q5_K_S | 5 | 28.77 GB | 31.27 GB | New k-quant method. Uses GGML_TYPE_Q5_K for all tensors |
| falcon40b-instruct.ggccv1.q6_K.bin | q6_K | 6 | 34.33 GB | 36.83 GB | New k-quant method. Uses GGML_TYPE_Q8_K - 6-bit quantization - for all tensors |
| falcon40b-instruct.ggccv1.q8_0.bin | q8_0 | 8 | 44.46 GB | 46.96 GB | Original llama.cpp quant method, 8-bit. Almost indistinguishable from float16. High resource use and slow. Not recommended for most users. |

**Note**: the above RAM figures assume no GPU offloading. If layers are offloaded to the GPU, this will reduce RAM usage and use VRAM instead.

<!-- footer start -->
## Discord

For further support, and discussions on these models and AI in general, join us at:

[TheBloke AI's Discord server](https://discord.gg/theblokeai)

## Thanks, and how to contribute.

Thanks to the [chirper.ai](https://chirper.ai) team!

I've had a lot of people ask if they can contribute. I enjoy providing models and helping people, and would love to be able to spend even more time doing it, as well as expanding into new projects like fine tuning/training.

If you're able and willing to contribute it will be most gratefully received and will help me to keep providing more models, and to start work on new AI projects.

Donaters will get priority support on any and all AI/LLM/model questions and requests, access to a private Discord room, plus other benefits.

* Patreon: https://patreon.com/TheBlokeAI
* Ko-Fi: https://ko-fi.com/TheBlokeAI

**Special thanks to**: Luke from CarbonQuill, Aemon Algiz, Dmitriy Samsonov.

**Patreon special mentions**: Spiking Neurons AB, Kevin Schuppel, Cory Kujawski, senxiiz, Luke Pendergrass, John Villwock, Ghost , Alex , Sean Connelly, Space Cruiser, Eugene Pentland, Pyrater, Matthew Berman, Dave, Derek Yates, Jonathan Leane, Viktor Bowallius, Michael Levine, Joseph William Delisle, Fred von Graf, Asp the Wyvern, Nikolai Manek, Pierre Kircher, webtim, K, RoA, Karl Bernard, Artur Olbinski, Rainer Wilmers, Ai Maven, Nathan LeClaire, Ajan Kanaga, Stephen Murray, Edmond Seymore, zynix , Imad Khwaja, John Detwiler, Randy H, subjectnull, Alps Aficionado, Greatston Gnanesh, Trenton Dambrowitz, Junyu Yang, Raven Klaugh, biorpg, Deep Realms, vamX, Talal Aujan, Johann-Peter Hartmann, WelcomeToTheClub, Chris McCloskey, Luke, chris gileta, terasurfer , Iucharbius , Preetika Verma, Willem Michiel, Fen Risland, SuperWojo, Khalefa Al-Ahmad, Daniel P. Andersen, Gabriel Puliatti, Illia Dulskyi, Willian Hasse, Oscar Rangel, ya boyyy, Mano Prime, Lone Striker, Kalila

Thank you to all my generous patrons and donaters!

<!-- footer end -->

# Original model card: Falcon 40B-Instruct GGML


# ✨ Falcon-40B-Instruct

**Falcon-40B-Instruct is a 40B parameters causal decoder-only model built by [TII](https://www.tii.ae) based on [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b) and finetuned on a mixture of [Baize](https://github.com/project-baize/baize-chatbot). It is made available under the Apache 2.0 license.**

*Paper coming soon 😊.*

🤗 To get started with Falcon (inference, finetuning, quantization, etc.), we recommend reading [this great blogpost fron HF](https://huggingface.co/blog/falcon)!

## Why use Falcon-40B-Instruct?

* **You are looking for a ready-to-use chat/instruct model based on [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b).**
* **Falcon-40B is the best open-source model available.** It outperforms [LLaMA](https://github.com/facebookresearch/llama), [StableLM](https://github.com/Stability-AI/StableLM), [RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1), [MPT](https://huggingface.co/mosaicml/mpt-7b), etc. See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
* **It features an architecture optimized for inference**, with FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) and multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)). 

💬 **This is an instruct model, which may not be ideal for further finetuning.** If you are interested in building your own instruct/chat model, we recommend starting from [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b). 

💸 **Looking for a smaller, less expensive model?** [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) is Falcon-40B-Instruct's little brother!

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```

For fast inference with Falcon, check-out [Text Generation Inference](https://github.com/huggingface/text-generation-inference)! Read more in this [blogpost]((https://huggingface.co/blog/falcon). 

You will need **at least 85-100GB of memory** to swiftly run inference with Falcon-40B.



# Model Card for Falcon-40B-Instruct

## Model Details

### Model Description

- **Developed by:** [https://www.tii.ae](https://www.tii.ae);
- **Model type:** Causal decoder-only;
- **Language(s) (NLP):** English and French;
- **License:** Apache 2.0;
- **Finetuned from model:** [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b).

### Model Source

- **Paper:** *coming soon*.

## Uses

### Direct Use

Falcon-40B-Instruct has been finetuned on a chat dataset.

### Out-of-Scope Use

Production use without adequate assessment of risks and mitigation; any use cases which may be considered irresponsible or harmful. 

## Bias, Risks, and Limitations

Falcon-40B-Instruct is mostly trained on English data, and will not generalize appropriately to other languages. Furthermore, as it is trained on a large-scale corpora representative of the web, it will carry the stereotypes and biases commonly encountered online.

### Recommendations

We recommend users of Falcon-40B-Instruct to develop guardrails and to take appropriate precautions for any production use.

## How to Get Started with the Model


```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```

## Training Details

### Training Data

Falcon-40B-Instruct was finetuned on a 150M tokens from [Bai ze](https://github.com/project-baize/baize-chatbot) mixed with 5% of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) data. 


The data was tokenized with the Falcon-[7B](https://huggingface.co/tiiuae/falcon-7b)/[40B](https://huggingface.co/tiiuae/falcon-40b) tokenizer.


## Evaluation

*Paper coming soon.*

See the [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for early results.


## Technical Specifications 

For more information about pretraining, see [Falcon-40B](https://huggingface.co/tiiuae/falcon-40b).

### Model Architecture and Objective

Falcon-40B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is broadly adapted from the GPT-3 paper ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)), with the following differences:

* **Positionnal embeddings:** rotary ([Su et al., 2021](https://arxiv.org/abs/2104.09864));
* **Attention:** multiquery ([Shazeer et al., 2019](https://arxiv.org/abs/1911.02150)) and FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135));
* **Decoder-block:** parallel attention/MLP with a single layer norm.

For multiquery, we are using an internal variant which uses independent key and values per tensor parallel degree.

| **Hyperparameter** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| Layers             | 60        |                                        |
| `d_model`          | 8192      |                                        |
| `head_dim`         | 64        | Reduced to optimise for FlashAttention |
| Vocabulary         | 65024     |                                        |
| Sequence length    | 2048      |                                        |

### Compute Infrastructure

#### Hardware

Falcon-40B-Instruct was trained on AWS SageMaker, on 64 A100 40GB GPUs in P4d instances. 

#### Software

Falcon-40B-Instruct was trained a custom distributed training codebase, Gigatron. It uses a 3D parallelism approach combined with ZeRO and high-performance Triton kernels (FlashAttention, etc.)


## Citation

*Paper coming soon* 😊. In the meanwhile, you can use the following information to cite: 
```
@article{falcon40b,
  title={{Falcon-40B}: an open large language model with state-of-the-art performance},
  author={Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and Cappelli, Alessandro and Cojocaru, Ruxandra and Debbah, Merouane and Goffinet, Etienne and Heslow, Daniel and Launay, Julien and Malartic, Quentin and Noune, Badreddine and Pannier, Baptiste and Penedo, Guilherme},
  year={2023}
}
```

To learn more about the pretraining dataset, see the 📓 [RefinedWeb paper](https://arxiv.org/abs/2306.01116).

```
@article{refinedweb,
  title={The {R}efined{W}eb dataset for {F}alcon {LLM}: outperforming curated corpora with web data, and web data only},
  author={Guilherme Penedo and Quentin Malartic and Daniel Hesslow and Ruxandra Cojocaru and Alessandro Cappelli and Hamza Alobeidli and Baptiste Pannier and Ebtesam Almazrouei and Julien Launay},
  journal={arXiv preprint arXiv:2306.01116},
  eprint={2306.01116},
  eprinttype = {arXiv},
  url={https://arxiv.org/abs/2306.01116},
  year={2023}
}
```

To cite the [Baize](https://github.com/project-baize/baize-chatbot) instruction dataset used for this model: 
```
@article{xu2023baize,
  title={Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data},
  author={Xu, Canwen and Guo, Daya and Duan, Nan and McAuley, Julian},
  journal={arXiv preprint arXiv:2304.01196},
  year={2023}
}
```


## License

Falcon-40B-Instruct is made available under the Apache 2.0 license.

## Contact
falconllm@tii.ae
