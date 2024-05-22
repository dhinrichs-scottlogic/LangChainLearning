import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import DirectoryLoader

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# single prompt
response = llm.invoke("Hello, how are you?")
print(response)

# multiple prompts
response = llm.batch(["Hello, how are you?", "Write a poem about AI"])
print(response)

# receive response in chunks
response = llm.stream("Write a poem about AI")
for chunk in response:
    print(chunk.content, end="", flush=True)


# load files from a directory
DATA_PATH = "data"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents
