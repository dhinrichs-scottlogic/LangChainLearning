import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader


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


# load text files from a directory
loader = DirectoryLoader('./data', glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()
print(documents[0].page_content[:100])

