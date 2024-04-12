# Import necessary libraries/modules

import requests
import json
import sys

API_URL = "https://a100-1.alta3.training"

# Initialize a list called CHAT to store the conversation history.
CHAT = [
]

# Define the initial instruction for the conversation.
with open('/home/student/mycode/receptionist/receptionist-beta.txt', 'r') as file:
    character_context = file.read()

INSTRUCTION = character_context

### everything below this point is the same
def trim(s):
    return s.strip()

def trim_trailing(s):
    return s.rstrip()

def format_prompt(s):
    return f"{INSTRUCTION}{CHAT}\n### Customer: {s}\n### Fine Hair Salon: "

def chat_completion(question):
    prompt = trim_trailing(format_prompt(question))
    data = json.dumps({
        "prompt": prompt,
        "temperature": 0.7,
        # "top_k": 40,
        "top_p": 0.9,
        "mirostat": 2,
        "n_predict": 512,
        "repeat_penalty": 1.18,
        "stop": ["Customer:"],
        "stream": True
    })

    answer = ''
    response = requests.post(
        f"{API_URL}/completion",
        headers={"Content-Type": "application/json"},
        data=data,
        stream=True
    )
    print("Fine Hair Salon: ", end="")
    for line in response.iter_lines(decode_unicode=True):
         if line.startswith("data: "):
             data_str = line[5:]  # Remove "data: " prefix
             content = json.loads(data_str)["content"]
             answer += content
             sys.stdout.write(content)
             sys.stdout.flush()

    print()

    CHAT.extend([question, trim(answer)])

chat_completion(INSTRUCTION)
while True:
    question = input("Customer: ")
    chat_completion(question)
