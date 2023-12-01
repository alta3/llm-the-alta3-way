from llama_cpp import Llama

llm = Llama(model_path="../../model/Llama-2-70B-Orca-200k/model/llama-2-70b-orca-200k.Q3_K_S.gguf", n_gpu_layers=80,  n_gqa=8, verbose=False)

# Specify the file path
file_path = "../llama.cpp/prompts/receptionist.txt"

# Read the contents of the file
with open(file_path, 'r') as file:
    prompt = file.read()

output = llm(prompt, max_tokens=0)

# display the response in a cleaner fashion
print("Prompt: " + prompt)
print(output["choices"][0]["text"])
