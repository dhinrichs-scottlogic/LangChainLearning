import subprocess
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.evaluation import load_evaluator

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
with open("./text/output.txt", "r") as file:
    output_text = file.read().strip()

with open("./text/input.txt", "r") as file:
    input_text = file.read().strip()

# evaluator example
evaluator = load_evaluator("pairwise_embedding_distance")
# x = evaluator.evaluate_string_pairs(prediction=input_text, prediction_b=output_text)
# print(x["score"])

# 0.1867 for a painting of a blue bowl and two cups vs abstract art


# run textToImage.py
subprocess.run(["python", "textToImage.py"])

# run imageToText.py
subprocess.run(["python", "imageToText.py"])

# compare input and output
x = evaluator.evaluate_string_pairs(prediction=input_text, prediction_b=output_text)
print(x["score"])

# if x is less than 0.15, add "as abstract art" to the input text
if x["score"] < 0.15:
    with open("./text/input.txt", "w") as f:
        f.write(input_text + " as abstract art")

# if x is less than 0.5, add "as abstract art" to the input text
if x["score"] < 0.1:
    with open("./text/input.txt", "w") as f:
        f.write(input_text + " underwater")

# otherwise save output text to input.txt
else:
    
    with open("./text/input.txt", "w") as f:
        f.write(output_text)
