import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')


API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
headers = {"Authorization": "Bearer " + api_key}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("image.jpg")

print(output)

with open("./text/output.txt", "w") as f:
    f.write((output[0]["generated_text"]))