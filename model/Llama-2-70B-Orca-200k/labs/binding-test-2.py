from llama_cpp import Llama

LLM = Llama(model_path="../../model/Llama-2-70B-Orca-200k/model/llama-2-70b-orca-200k.Q3_K_S.gguf", n_gpu_layers=80)

prompt = "Question: Can you tell me about the largest city in Europe? Answer:"

# set max_tokens to 0 to remove the response size limit
output = LLM(prompt, max_tokens=0)

# display the response in a cleaner fashion
print("Prompt: " + prompt)
print(output["choices"][0]["text"])
