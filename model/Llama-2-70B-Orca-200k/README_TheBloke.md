---
inference: false
language:
- en
license: llama2
model_creator: ddobokki
model_link: https://huggingface.co/ddobokki/Llama-2-70b-orca-200k
model_name: Llama 2 70B Orca 200k
model_type: llama
pipeline_tag: text-generation
quantized_by: TheBloke
tags:
- llama-2
- instruct
- instruction
---

<!-- header start -->
<!-- 200823 -->
<div style="width: auto; margin-left: auto; margin-right: auto">
<img src="https://i.imgur.com/EBdldam.jpg" alt="TheBlokeAI" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="display: flex; flex-direction: column; align-items: flex-start;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://discord.gg/theblokeai">Chat & support: TheBloke's Discord server</a></p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://www.patreon.com/TheBlokeAI">Want to contribute? TheBloke's Patreon page</a></p>
    </div>
</div>
<div style="text-align:center; margin-top: 0em; margin-bottom: 0em"><p style="margin-top: 0.25em; margin-bottom: 0em;">TheBloke's LLM work is generously supported by a grant from <a href="https://a16z.com">andreessen horowitz (a16z)</a></p></div>
<hr style="margin-top: 1.0em; margin-bottom: 1.0em;">
<!-- header end -->

# Llama 2 70B Orca 200k - GGUF
- Model creator: [ddobokki](https://huggingface.co/ddobokki)
- Original model: [Llama 2 70B Orca 200k](https://huggingface.co/ddobokki/Llama-2-70b-orca-200k)

## Description

This repo contains GGUF format model files for [ddobokki's Llama 2 70B Orca 200k](https://huggingface.co/ddobokki/Llama-2-70b-orca-200k).

<!-- README_GGUF.md-about-gguf start -->
### About GGUF

GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp.

The key benefit of GGUF is that it is a extensible, future-proof format which stores more information about the model as metadata. It also includes significantly improved tokenization code, including for the first time full support for special tokens. This should improve performance, especially with models that use new special tokens and implement custom prompt templates.

Here are a list of clients and libraries that are known to support GGUF:
* [llama.cpp](https://github.com/ggerganov/llama.cpp).
* [text-generation-webui](https://github.com/oobabooga/text-generation-webui), the most widely used web UI, with many features and powerful extensions.
* [KoboldCpp](https://github.com/LostRuins/koboldcpp), a fully featured web UI, with full GPU accel across multiple platforms and GPU architectures. Especially good for story telling.
* [LM Studio](https://lmstudio.ai/), an easy-to-use and powerful local GUI with GPU acceleration on both Windows (NVidia and AMD), and macOS.
* [LoLLMS Web UI](https://github.com/ParisNeo/lollms-webui), a great web UI with many interesting and unique features, including a full model library for easy model selection.
* [ctransformers](https://github.com/marella/ctransformers), a Python library with GPU accel, LangChain support, and OpenAI-compatible AI server.
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), a Python library with GPU accel, LangChain support, and OpenAI-compatible API server.
* [candle](https://github.com/huggingface/candle), a Rust ML framework with a focus on performance, including GPU support, and ease of use.

<!-- README_GGUF.md-about-gguf end -->
<!-- repositories-available start -->
## Repositories available

* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF)
* [2, 3, 4, 5, 6 and 8-bit GGML models for CPU+GPU inference (deprecated)](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGML)
* [ddobokki's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/ddobokki/Llama-2-70b-orca-200k)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: Guanaco

```
### Human: {prompt}
### Assistant:

```

<!-- prompt-template end -->
<!-- compatibility_gguf start -->
## Compatibility

These quantised GGUF files are compatible with llama.cpp from August 21st 2023 onwards, as of commit [6381d4e110bd0ec02843a60bbeb8b6fc37a9ace9](https://github.com/ggerganov/llama.cpp/commit/6381d4e110bd0ec02843a60bbeb8b6fc37a9ace9)

They are now also compatible with many third party UIs and libraries - please see the list at the top of the README.

## Explanation of quantisation methods
<details>
  <summary>Click to see details</summary>

The new methods available are:
* GGML_TYPE_Q2_K - "type-1" 2-bit quantization in super-blocks containing 16 blocks, each block having 16 weight. Block scales and mins are quantized with 4 bits. This ends up effectively using 2.5625 bits per weight (bpw)
* GGML_TYPE_Q3_K - "type-0" 3-bit quantization in super-blocks containing 16 blocks, each block having 16 weights. Scales are quantized with 6 bits. This end up using 3.4375 bpw.
* GGML_TYPE_Q4_K - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using 4.5 bpw.
* GGML_TYPE_Q5_K - "type-1" 5-bit quantization. Same super-block structure as GGML_TYPE_Q4_K resulting in 5.5 bpw
* GGML_TYPE_Q6_K - "type-0" 6-bit quantization. Super-blocks with 16 blocks, each block having 16 weights. Scales are quantized with 8 bits. This ends up using 6.5625 bpw

Refer to the Provided Files table below to see what files use which methods, and how.
</details>
<!-- compatibility_gguf end -->

<!-- README_GGUF.md-provided-files start -->
## Provided files

| Name | Quant method | Bits | Size | Max RAM required | Use case |
| ---- | ---- | ---- | ---- | ---- | ----- |
| [llama-2-70b-orca-200k.Q2_K.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q2_K.gguf) | Q2_K | 2 | 29.28 GB| 31.78 GB | smallest, significant quality loss - not recommended for most purposes |
| [llama-2-70b-orca-200k.Q3_K_S.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q3_K_S.gguf) | Q3_K_S | 3 | 29.92 GB| 32.42 GB | very small, high quality loss |
| [llama-2-70b-orca-200k.Q3_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q3_K_M.gguf) | Q3_K_M | 3 | 33.19 GB| 35.69 GB | very small, high quality loss |
| [llama-2-70b-orca-200k.Q3_K_L.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q3_K_L.gguf) | Q3_K_L | 3 | 36.15 GB| 38.65 GB | small, substantial quality loss |
| [llama-2-70b-orca-200k.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q4_0.gguf) | Q4_0 | 4 | 38.87 GB| 41.37 GB | legacy; small, very high quality loss - prefer using Q3_K_M |
| [llama-2-70b-orca-200k.Q4_K_S.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q4_K_S.gguf) | Q4_K_S | 4 | 39.07 GB| 41.57 GB | small, greater quality loss |
| [llama-2-70b-orca-200k.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q4_K_M.gguf) | Q4_K_M | 4 | 41.42 GB| 43.92 GB | medium, balanced quality - recommended |
| [llama-2-70b-orca-200k.Q5_0.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q5_0.gguf) | Q5_0 | 5 | 47.46 GB| 49.96 GB | legacy; medium, balanced quality - prefer using Q4_K_M |
| [llama-2-70b-orca-200k.Q5_K_S.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q5_K_S.gguf) | Q5_K_S | 5 | 47.46 GB| 49.96 GB | large, low quality loss - recommended |
| [llama-2-70b-orca-200k.Q5_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF/blob/main/llama-2-70b-orca-200k.Q5_K_M.gguf) | Q5_K_M | 5 | 48.75 GB| 51.25 GB | large, very low quality loss - recommended |
| llama-2-70b-orca-200k.Q6_K.gguf | Q6_K | 6 | 56.59 GB| 59.09 GB | very large, extremely low quality loss |
| llama-2-70b-orca-200k.Q8_0.gguf | Q8_0 | 8 | 73.29 GB| 75.79 GB | very large, extremely low quality loss - not recommended |

**Note**: the above RAM figures assume no GPU offloading. If layers are offloaded to the GPU, this will reduce RAM usage and use VRAM instead.

### Q6_K and Q8_0 files are split and require joining

**Note:** HF does not support uploading files larger than 50GB. Therefore I have uploaded the Q6_K and Q8_0 files as split files.

<details>
  <summary>Click for instructions regarding Q6_K and Q8_0 files</summary>
   
### q6_K 
Please download:
* `llama-2-70b-orca-200k.Q6_K.gguf-split-a`
* `llama-2-70b-orca-200k.Q6_K.gguf-split-b`

### q8_0
Please download:
* `llama-2-70b-orca-200k.Q8_0.gguf-split-a`
* `llama-2-70b-orca-200k.Q8_0.gguf-split-b`

To join the files, do the following:

Linux and macOS:
```
cat llama-2-70b-orca-200k.Q6_K.gguf-split-* > llama-2-70b-orca-200k.Q6_K.gguf && rm llama-2-70b-orca-200k.Q6_K.gguf-split-*
cat llama-2-70b-orca-200k.Q8_0.gguf-split-* > llama-2-70b-orca-200k.Q8_0.gguf && rm llama-2-70b-orca-200k.Q8_0.gguf-split-*
```
Windows command line:
```
COPY /B llama-2-70b-orca-200k.Q6_K.gguf-split-a + llama-2-70b-orca-200k.Q6_K.gguf-split-b llama-2-70b-orca-200k.Q6_K.gguf
del llama-2-70b-orca-200k.Q6_K.gguf-split-a llama-2-70b-orca-200k.Q6_K.gguf-split-b

COPY /B llama-2-70b-orca-200k.Q8_0.gguf-split-a + llama-2-70b-orca-200k.Q8_0.gguf-split-b llama-2-70b-orca-200k.Q8_0.gguf
del llama-2-70b-orca-200k.Q8_0.gguf-split-a llama-2-70b-orca-200k.Q8_0.gguf-split-b
```

</details>
<!-- README_GGUF.md-provided-files end -->

<!-- README_GGUF.md-how-to-run start -->
## Example `llama.cpp` command

Make sure you are using `llama.cpp` from commit [6381d4e110bd0ec02843a60bbeb8b6fc37a9ace9](https://github.com/ggerganov/llama.cpp/commit/6381d4e110bd0ec02843a60bbeb8b6fc37a9ace9) or later.

For compatibility with older versions of llama.cpp, or for any third-party libraries or clients that haven't yet updated for GGUF, please use GGML files instead.

```
./main -t 10 -ngl 32 -m llama-2-70b-orca-200k.q4_K_M.gguf --color -c 4096 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "### Human: {prompt}\n### Assistant:"
```
Change `-t 10` to the number of physical CPU cores you have. For example if your system has 8 cores/16 threads, use `-t 8`. If offloading all layers to GPU, set `-t 1`.

Change `-ngl 32` to the number of layers to offload to GPU. Remove it if you don't have GPU acceleration.

Change `-c 4096` to the desired sequence length for this model. For extended sequence models - eg 8K, 16K, 32K - the necessary RoPE scaling parameters are read from the GGUF file and set by llama.cpp automatically.

If you want to have a chat-style conversation, replace the `-p <PROMPT>` argument with `-i -ins`

For other parameters and how to use them, please refer to [the llama.cpp documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)

## How to run in `text-generation-webui`

Further instructions here: [text-generation-webui/docs/llama.cpp.md](https://github.com/oobabooga/text-generation-webui/blob/main/docs/llama.cpp.md).

## How to run from Python code

You can use GGUF models from Python using the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) or [ctransformers](https://github.com/marella/ctransformers) libraries.

### How to load this model from Python using ctransformers

#### First install the package

```bash
# Base ctransformers with no GPU acceleration
pip install ctransformers>=0.2.24
# Or with CUDA GPU acceleration
pip install ctransformers[cuda]>=0.2.24
# Or with ROCm GPU acceleration
CT_HIPBLAS=1 pip install ctransformers>=0.2.24 --no-binary ctransformers
# Or with Metal GPU acceleration for macOS systems
CT_METAL=1 pip install ctransformers>=0.2.24 --no-binary ctransformers
```

#### Simple example code to load one of these GGUF models

```python
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-70B-Orca-200k-GGUF", model_file="llama-2-70b-orca-200k.q4_K_M.gguf", model_type="llama", gpu_layers=50)

print(llm("AI is going to"))
```

## How to use with LangChain

Here's guides on using llama-cpp-python or ctransformers with LangChain:

* [LangChain + llama-cpp-python](https://python.langchain.com/docs/integrations/llms/llamacpp)
* [LangChain + ctransformers](https://python.langchain.com/docs/integrations/providers/ctransformers)

<!-- README_GGUF.md-how-to-run end -->

<!-- footer start -->
<!-- 200823 -->
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

**Special thanks to**: Aemon Algiz.

**Patreon special mentions**: Russ Johnson, J, alfie_i, Alex, NimbleBox.ai, Chadd, Mandus, Nikolai Manek, Ken Nordquist, ya boyyy, Illia Dulskyi, Viktor Bowallius, vamX, Iucharbius, zynix, Magnesian, Clay Pascal, Pierre Kircher, Enrico Ros, Tony Hughes, Elle, Andrey, knownsqashed, Deep Realms, Jerry Meng, Lone Striker, Derek Yates, Pyrater, Mesiah Bishop, James Bentley, Femi Adebogun, Brandon Frisco, SuperWojo, Alps Aficionado, Michael Dempsey, Vitor Caleffi, Will Dee, Edmond Seymore, usrbinkat, LangChain4j, Kacper Wikieł, Luke Pendergrass, John Detwiler, theTransient, Nathan LeClaire, Tiffany J. Kim, biorpg, Eugene Pentland, Stanislav Ovsiannikov, Fred von Graf, terasurfer, Kalila, Dan Guido, Nitin Borwankar, 阿明, Ai Maven, John Villwock, Gabriel Puliatti, Stephen Murray, Asp the Wyvern, danny, Chris Smitley, ReadyPlayerEmma, S_X, Daniel P. Andersen, Olakabola, Jeffrey Morgan, Imad Khwaja, Caitlyn Gatomon, webtim, Alicia Loh, Trenton Dambrowitz, Swaroop Kallakuri, Erik Bjäreholt, Leonard Tan, Spiking Neurons AB, Luke @flexchar, Ajan Kanaga, Thomas Belote, Deo Leter, RoA, Willem Michiel, transmissions 11, subjectnull, Matthew Berman, Joseph William Delisle, David Ziegler, Michael Davis, Johann-Peter Hartmann, Talal Aujan, senxiiz, Artur Olbinski, Rainer Wilmers, Spencer Kim, Fen Risland, Cap'n Zoog, Rishabh Srivastava, Michael Levine, Geoffrey Montalvo, Sean Connelly, Alexandros Triantafyllidis, Pieter, Gabriel Tamborski, Sam, Subspace Studios, Junyu Yang, Pedro Madruga, Vadim, Cory Kujawski, K, Raven Klaugh, Randy H, Mano Prime, Sebastain Graf, Space Cruiser


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

<!-- original-model-card start -->
# Original model card: ddobokki's Llama 2 70B Orca 200k


# Llama-2-70b-orca-200k model card

### Used Datasets
- OpenOrca (200k sampling)

### Prompt Template
```
### Human: {Human}
### Assistant: {Assistant}
```

### Contribute
[ddobokki](https://github.com/ddobokki)

[YooSungHyun](https://github.com/YooSungHyun)

### License
[LICENSE.txt](meta-license/LICENSE.txt)

### USE_POLICY
[USE_POLICY.md](meta-license/USE_POLICY.md)

### Responsible Use Guide
[Responsible-Use-Guide.pdf](meta-license/Responsible-Use-Guide.pdf)

<!-- original-model-card end -->
