# Import necessary libraries/modules
import requests
import json
import sys

# Define the API URL for the AI service.
API_URL = "https://a100-88.alta3.training"

# Initialize a list called CHAT to store the conversation history.
CHAT = [
    "Hello, Assistant.",
    "Hello. How may I help you today?",
    "Please tell me the largest city in Europe.",
    "Sure. The largest city in Europe is Moscow, the capital of Russia."
]

# Define the initial instruction for the conversation.
INSTRUCTION = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, entertaining, detailed, and succinct answers to the human's questions."

# Define some utility functions for removing whitespaces and designing the prompt to contain all of the known conversation.
def trim(s):
    return s.strip()

def trim_trailing(s):
    return s.rstrip()

def format_prompt(s):
    return f"{INSTRUCTION}{CHAT}\n### Human: {s}\n### Assistant: "

# Define a function for the chat completion process.
def chat_completion(question):
    # Prepare a prompt by formatting the user's question and conversation history.
    prompt = trim_trailing(format_prompt(question))

    # Configure options for generating a response using the AI model.
    data = json.dumps({
        "prompt": prompt,
        "temperature": 0.7,
        # "top_k": 40,
        "top_p": 0.9,
        "mirostat": 2,
        "n_predict": 512,
        "repeat_penalty": 1.18,
        "stop": ["User:"],
        "stream": True
    })

    # Initialize variables to store the answer and response from the AI model.
    answer = ''
    response = requests.post(
        f"{API_URL}/completion",
        headers={"Content-Type": "application/json"},
        data=data,
        stream=True
    )

    # Process and display the response from the AI model.
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data_str = line[5:]  # Remove "data: " prefix
            content = json.loads(data_str)["content"]
            answer += content
            sys.stdout.write(content)
            sys.stdout.flush()

    # Print a new line to separate the user's input and the assistant's response.
    print()

    # Add the user's question and trimmed answer to the conversation history.
    CHAT.extend([question, trim(answer)])

# Start the conversation loop. The user can ask questions and get responses.
while True:
    question = input("> ")
    chat_completion(question)
