# Import necessary libraries/modules

import os
import requests
import json
import sys

API_URL = "https://a100-1.alta3.training"

CHAT = [
]

with open('/home/student/mycode/receptionist/receptionist-json.txt', 'r') as file:
    character_context = file.read()

INSTRUCTION = character_context

def trim(s):
    return s.strip()

def trim_trailing(s):
    return s.rstrip()

def format_prompt(s):
    return f"{INSTRUCTION}{CHAT}\n Customer: {s}\n Fine Hair Salon: "

def set_appointment():
    print("Thank you for using Salon Chat, your appointment has been created. We'll see you soon!")
    json_prompt= f"{INSTRUCTION}{CHAT}\n Customer: Provide the JSON.\n Fine Hair Salon: "
    json_data = json.dumps({
        "prompt": json_prompt,
        "temperature": 0.7,
        # "top_k": 40,
        "top_p": 0.9,
        "mirostat": 2,
        "n_predict": 512,
        "repeat_penalty": 1.18,
        "stop": ["Customer:"],
        "stream": True
        })
    json_response = requests.post(
        f"{API_URL}/completion",
        headers={"Content-Type": "application/json"},
        data=json_data,
        stream=True
        )
    json_content = ""
    for line in json_response.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data_str = line[5:]  # Remove "data: " prefix
            json_content += json.loads(data_str)["content"]

    file_path = os.path.join("./appointments", f"appointments.txt")

    if os.path.exists(file_path):
        # If the file exists, open it in append mode and add a comma before appending the new content
        with open(file_path, 'a') as txt_file:
            txt_file.write("," + json_content)
    else:
        # If the file doesn't exist, create it and write the JSON content
        with open(file_path, 'w') as txt_file:
            txt_file.write(json_content)

    sys.exit(0)

def chat_completion(question):
    prompt = trim_trailing(format_prompt(question))
    data = json.dumps({
        "prompt": prompt,
        "temperature": 0.7,
        # "top_k": 40,
        "top_p": 0.9,
        "mirostat": 2,
        "n_predict": 512,
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
            if "JSON" in content:
                 set_appointment()

            answer += content
            sys.stdout.write(content)
            sys.stdout.flush()

    print()

    CHAT.extend([question, trim(answer)])

chat_completion(INSTRUCTION)
while True:
    question = input("Customer: ")
    chat_completion(question)
