from ctransformers import AutoModelForCausalLM, AutoTokenizer

model_bin = "/home/ubuntu/llm/model/orca_mini_v3_13b/orca_mini_v3_13b.gguf.q8_0.bin"

model = AutoModelForCausalLM.from_pretrained(model_bin, model_type='llama')
tokenizer = AutoTokenizer.from_pretrained(model)

system_prompt = "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n"

#generate text steps
instruction = "Tell me about Orcas."
prompt = f"{system_prompt}### User: {instruction}\n\n### Assistant:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
del inputs['token_type_ids']
output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=4096)

print(tokenizer.decode(output[0], skip_special_tokens=True))
