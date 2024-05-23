import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": f"Bearer {api_key}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
with open("./text/output.txt", "r") as file:
	input_text = file.read().strip()
    
output = query({
    "inputs": input_text,
})[0]["generated_text"]

outputWithoutPrompt = output.split(input_text)[1].strip()

print(output)

with open("./text/textGenerationOutput.txt", "w") as f:
    f.write(outputWithoutPrompt)
