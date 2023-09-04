---
inference: false
language:
- en
library_name: transformers
license: other
model_creator: Pankaj Mathur
model_link: https://huggingface.co/psmathur/orca_mini_v3_13b
model_name: Orca Mini v3 13B
model_type: llama
quantized_by: TheBloke
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

# Orca Mini v3 13B - GGML
- Model creator: [Pankaj Mathur](https://huggingface.co/psmathur)
- Original model: [Orca Mini v3 13B](https://huggingface.co/psmathur/orca_mini_v3_13b)

## Description

This repo contains GGML format model files for [Pankaj Mathur's Orca Mini v3 13B](https://huggingface.co/psmathur/orca_mini_v3_13b).

GGML files are for CPU + GPU inference using [llama.cpp](https://github.com/ggerganov/llama.cpp) and libraries and UIs which support this format, such as:
* [text-generation-webui](https://github.com/oobabooga/text-generation-webui), the most popular web UI. Supports NVidia CUDA GPU acceleration.
* [KoboldCpp](https://github.com/LostRuins/koboldcpp), a powerful GGML web UI with GPU acceleration on all platforms (CUDA and OpenCL). Especially good for story telling.
* [LM Studio](https://lmstudio.ai/), a fully featured local GUI with GPU acceleration on both Windows (NVidia and AMD), and macOS.
* [LoLLMS Web UI](https://github.com/ParisNeo/lollms-webui), a great web UI with CUDA GPU acceleration via the c_transformers backend.
* [ctransformers](https://github.com/marella/ctransformers), a Python library with GPU accel, LangChain support, and OpenAI-compatible AI server.
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), a Python library with GPU accel, LangChain support, and OpenAI-compatible API server.

## Repositories available

* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/orca_mini_v3_13B-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGML models for CPU+GPU inference](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML)
* [Pankaj Mathur's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/psmathur/orca_mini_v3_13b)

## Prompt template: orca_mini

```
### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.

### User:
{prompt}

### Input:
{input}

### Response:
```

<!-- compatibility_ggml start -->
## Compatibility

These quantised GGML files are compatible with llama.cpp as of June 6th, commit `2d43387`.

They should also be compatible with all UIs, libraries and utilities which use GGML.

## Explanation of the new k-quant methods
<details>
  <summary>Click to see details</summary>

The new methods available are:
* GGML_TYPE_Q2_K - "type-1" 2-bit quantization in super-blocks containing 16 blocks, each block having 16 weight. Block scales and mins are quantized with 4 bits. This ends up effectively using 2.5625 bits per weight (bpw)
* GGML_TYPE_Q3_K - "type-0" 3-bit quantization in super-blocks containing 16 blocks, each block having 16 weights. Scales are quantized with 6 bits. This end up using 3.4375 bpw.
* GGML_TYPE_Q4_K - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using 4.5 bpw.
* GGML_TYPE_Q5_K - "type-1" 5-bit quantization. Same super-block structure as GGML_TYPE_Q4_K resulting in 5.5 bpw
* GGML_TYPE_Q6_K - "type-0" 6-bit quantization. Super-blocks with 16 blocks, each block having 16 weights. Scales are quantized with 8 bits. This ends up using 6.5625 bpw
* GGML_TYPE_Q8_K - "type-0" 8-bit quantization. Only used for quantizing intermediate results. The difference to the existing Q8_0 is that the block size is 256. All 2-6 bit dot products are implemented for this quantization type.

Refer to the Provided Files table below to see what files use which methods, and how.
</details>
<!-- compatibility_ggml end -->

## Provided files

| Name | Quant method | Bits | Size | Max RAM required | Use case |
| ---- | ---- | ---- | ---- | ---- | ----- |
| [orca_mini_v3_13b.ggmlv3.q2_K.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q2_K.bin) | q2_K | 2 | 5.51 GB| 8.01 GB | New k-quant method. Uses GGML_TYPE_Q4_K for the attention.vw and feed_forward.w2 tensors, GGML_TYPE_Q2_K for the other tensors. |
| [orca_mini_v3_13b.ggmlv3.q3_K_L.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q3_K_L.bin) | q3_K_L | 3 | 6.93 GB| 9.43 GB | New k-quant method. Uses GGML_TYPE_Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else GGML_TYPE_Q3_K |
| [orca_mini_v3_13b.ggmlv3.q3_K_M.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q3_K_M.bin) | q3_K_M | 3 | 6.31 GB| 8.81 GB | New k-quant method. Uses GGML_TYPE_Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else GGML_TYPE_Q3_K |
| [orca_mini_v3_13b.ggmlv3.q3_K_S.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q3_K_S.bin) | q3_K_S | 3 | 5.66 GB| 8.16 GB | New k-quant method. Uses GGML_TYPE_Q3_K for all tensors |
| [orca_mini_v3_13b.ggmlv3.q4_0.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q4_0.bin) | q4_0 | 4 | 7.37 GB| 9.87 GB | Original quant method, 4-bit. |
| [orca_mini_v3_13b.ggmlv3.q4_1.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q4_1.bin) | q4_1 | 4 | 8.17 GB| 10.67 GB | Original quant method, 4-bit. Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models. |
| [orca_mini_v3_13b.ggmlv3.q4_K_M.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q4_K_M.bin) | q4_K_M | 4 | 7.87 GB| 10.37 GB | New k-quant method. Uses GGML_TYPE_Q6_K for half of the attention.wv and feed_forward.w2 tensors, else GGML_TYPE_Q4_K |
| [orca_mini_v3_13b.ggmlv3.q4_K_S.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q4_K_S.bin) | q4_K_S | 4 | 7.37 GB| 9.87 GB | New k-quant method. Uses GGML_TYPE_Q4_K for all tensors |
| [orca_mini_v3_13b.ggmlv3.q5_0.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q5_0.bin) | q5_0 | 5 | 8.97 GB| 11.47 GB | Original quant method, 5-bit. Higher accuracy, higher resource usage and slower inference. |
| [orca_mini_v3_13b.ggmlv3.q5_1.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q5_1.bin) | q5_1 | 5 | 9.78 GB| 12.28 GB | Original quant method, 5-bit. Even higher accuracy, resource usage and slower inference. |
| [orca_mini_v3_13b.ggmlv3.q5_K_M.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q5_K_M.bin) | q5_K_M | 5 | 9.23 GB| 11.73 GB | New k-quant method. Uses GGML_TYPE_Q6_K for half of the attention.wv and feed_forward.w2 tensors, else GGML_TYPE_Q5_K |
| [orca_mini_v3_13b.ggmlv3.q5_K_S.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q5_K_S.bin) | q5_K_S | 5 | 8.97 GB| 11.47 GB | New k-quant method. Uses GGML_TYPE_Q5_K for all tensors |
| [orca_mini_v3_13b.ggmlv3.q6_K.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q6_K.bin) | q6_K | 6 | 10.68 GB| 13.18 GB | New k-quant method. Uses GGML_TYPE_Q8_K for all tensors - 6-bit quantization |
| [orca_mini_v3_13b.ggmlv3.q8_0.bin](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML/blob/main/orca_mini_v3_13b.ggmlv3.q8_0.bin) | q8_0 | 8 | 13.79 GB| 16.29 GB | Original quant method, 8-bit. Almost indistinguishable from float16. High resource use and slow. Not recommended for most users. |

**Note**: the above RAM figures assume no GPU offloading. If layers are offloaded to the GPU, this will reduce RAM usage and use VRAM instead.

## How to run in `llama.cpp`

I use the following command line; adjust for your tastes and needs:

```
./main -t 10 -ngl 32 -m orca_mini_v3_13b.ggmlv3.q4_K_M.bin --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "### Instruction: Write a story about llamas\n### Response:"
```
Change `-t 10` to the number of physical CPU cores you have. For example if your system has 8 cores/16 threads, use `-t 8`.

Change `-ngl 32` to the number of layers to offload to GPU. Remove it if you don't have GPU acceleration.

Change `-c 2048` to the desired sequence length for this model. For example, `-c 4096` for a Llama 2 model.  For models that use RoPE, add `--rope-freq-base 10000 --rope-freq-scale 0.5` for doubled context, or `--rope-freq-base 10000 --rope-freq-scale 0.25` for 4x context.

If you want to have a chat-style conversation, replace the `-p <PROMPT>` argument with `-i -ins`

For other parameters and how to use them, please refer to [the llama.cpp documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)

## How to run in `text-generation-webui`

Further instructions here: [text-generation-webui/docs/llama.cpp-models.md](https://github.com/oobabooga/text-generation-webui/blob/main/docs/llama.cpp-models.md).

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

**Special thanks to**: Luke from CarbonQuill, Aemon Algiz.

**Patreon special mentions**: Willem Michiel, Ajan Kanaga, Cory Kujawski, Alps Aficionado, Nikolai Manek, Jonathan Leane, Stanislav Ovsiannikov, Michael Levine, Luke Pendergrass, Sid, K, Gabriel Tamborski, Clay Pascal, Kalila, William Sang, Will Dee, Pieter, Nathan LeClaire, ya boyyy, David Flickinger, vamX, Derek Yates, Fen Risland, Jeffrey Morgan, webtim, Daniel P. Andersen, Chadd, Edmond Seymore, Pyrater, Olusegun Samson, Lone Striker, biorpg, alfie_i, Mano Prime, Chris Smitley, Dave, zynix, Trenton Dambrowitz, Johann-Peter Hartmann, Magnesian, Spencer Kim, John Detwiler, Iucharbius, Gabriel Puliatti, LangChain4j, Luke @flexchar, Vadim, Rishabh Srivastava, Preetika Verma, Ai Maven, Femi Adebogun, WelcomeToTheClub, Leonard Tan, Imad Khwaja, Steven Wood, Stefan Sabev, Sebastain Graf, usrbinkat, Dan Guido, Sam, Eugene Pentland, Mandus, transmissions 11, Slarti, Karl Bernard, Spiking Neurons AB, Artur Olbinski, Joseph William Delisle, ReadyPlayerEmma, Olakabola, Asp the Wyvern, Space Cruiser, Matthew Berman, Randy H, subjectnull, danny, John Villwock, Illia Dulskyi, Rainer Wilmers, theTransient, Pierre Kircher, Alexandros Triantafyllidis, Viktor Bowallius, terasurfer, Deep Realms, SuperWojo, senxiiz, Oscar Rangel, Alex, Stephen Murray, Talal Aujan, Raven Klaugh, Sean Connelly, Raymond Fosdick, Fred von Graf, chris gileta, Junyu Yang, Elle


Thank you to all my generous patrons and donaters!

<!-- footer end -->

# Original model card: Pankaj Mathur's Orca Mini v3 13B


# orca_mini_v3_13b

A Llama2-13b model trained on Orca Style datasets.

**I am actively seeking sponsorship and partnership opportunities. If you're interested, please connect with me at www.linkedin.com/in/pankajam.**

## Evaluation

We evaluated orca_mini_v3_13b on a wide range of tasks using [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) from EleutherAI. 

Here are the results on metrics used by [HuggingFaceH4 Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

|||||
|:------:|:--------:|:-------:|:--------:|
|**Task**|**Metric**|**Value**|**Stderr**|
|*arc_challenge*|acc_norm|0.6314|0.0141|
|*hellaswag*|acc_norm|0.8242|0.0038|
|*mmlu*|acc_norm|0.5637|0.0351|
|*truthfulqa_mc*|mc2|0.5127|0.0157|
|**Total Average**|-|**0.6329877193**||


## Example Usage

Here is the prompt format

```
### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.

### User:
Tell me about Orcas.

### Assistant:

```

Below shows a code example on how to use this model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("psmathur/orca_mini_v3_13b")
model = AutoModelForCausalLM.from_pretrained(
  "psmathur/orca_mini_v3_13b",
  torch_dtype=torch.float16,
  load_in_8bit=True,
  low_cpu_mem_usage=True,
  device_map="auto"
)
system_prompt = "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n"

#generate text steps
instruction = "Tell me about Orcas."
prompt = f"{system_prompt}### User: {instruction}\n\n### Assistant:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=4096)

print(tokenizer.decode(output[0], skip_special_tokens=True))

```
#### Legal Disclaimer:

This model is bound by the usage restrictions of the original Llama-2 model. And comes with no warranty or gurantees of any kind.



#### Limitations & Biases:

While this model aims for accuracy, it can occasionally produce inaccurate or misleading results. 

Despite diligent efforts in refining the pretraining data, there remains a possibility for the generation of inappropriate, biased, or offensive content. 

Exercise caution and cross-check information when necessary.



### Citiation:

Please kindly cite using the following BibTeX:

```
@misc{orca_mini_v3_13b,
  author = {Pankaj Mathur},
  title = {orca_mini_v3_13b: An Orca Style Llama2-70b model},
  year = {2023},
  publisher = {HuggingFace},
  journal = {HuggingFace repository},
  howpublished = {\url{https://https://huggingface.co/psmathur/orca_mini_v3_13b},
}
```

```
@misc{mukherjee2023orca,
      title={Orca: Progressive Learning from Complex Explanation Traces of GPT-4}, 
      author={Subhabrata Mukherjee and Arindam Mitra and Ganesh Jawahar and Sahaj Agarwal and Hamid Palangi and Ahmed Awadallah},
      year={2023},
      eprint={2306.02707},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@software{touvron2023llama2,
  title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
  author={Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava,
 Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller,
Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez Madian Khabsa, Isabel Kloumann,
Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov,
Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith,
Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu , Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan,
Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom},
  year={2023}
}
```
