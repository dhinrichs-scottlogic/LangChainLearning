import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer " + api_key}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
with open("./text/input.txt", "r") as file:
	input_text = file.read().strip()

image_bytes = query({
	"inputs": input_text,
})

image = Image.open(io.BytesIO(image_bytes))
image.save("image.jpg")