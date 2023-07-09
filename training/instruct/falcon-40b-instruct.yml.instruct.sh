#!/bin/sh
cat << EOF
### Instruction:

Response output shall be formatted in Markdown.
Develop an ansible playbook that satisfies the provided input requirements.

Response should be written in three sections:
 1) High level overview of the expected ansible playbook tasks in plain english. Title: Overview
 2) Summary of expected playbook variables, example values, and description of thier intended use. Title: Variables
 3) The full ansible playbook. Title: Playbook

Additional ansible playbook requirements:
 - Never use the ansible shell module to accomplish tasks for which an ansible module exists

### Input:

Install a C++ LLM project onto a Ubuntu Linux target machine including:
 - installing dependencies
 - git cloning the source repo
 - compiling the application via make
 - downloading supporting datasets and configuration files
 - running a test command to verify compilation succedded

### Response:

## Overview
This ansible playbook installa a C++ LLM project (ggllm.cpp) 
and the Falcon 40B Instruct GGML model for Alpaca Instruction Following 
infrence tasks. 

Expected ansible tasks modules:
 - debug - verify all variables provided
 - package - install project dependencies via apt 
 - git - clone remote project repoistory into target system
 - shell - required for compilation step which has no purpose built ansible module 
 - get_url - fetch extra required files for successfull run of falcon_main
 - synchronize - deoply folders of content (pormpts) to the target system
 
## Variables
 - ggllm_url: source url for ggllm.cpp project git repo
 - ggllm_branch: source branch for ggllm.cpp project git repo 
 - ggllm_dir: destination location for ggllm.cpp install
 - llama_cublas: enable gpu asseleration (1 = enabled, 0 = disabled)
 - falcon_filename: filename of ggml formatted falcon-40b parameters
 - falcon_url: source url of ggml formatted falcon-40b parameters
 - falcon_tokenizer_url: source url of falcon-40b 

## Playbook

EOF

echo '```yaml'
cat ../falcon-40b-instruct.yml
echo '```'
