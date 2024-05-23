import os
import shutil
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

# CHROMA_PATH = "chroma"

# def main():
#     generate_data_store()

# def generate_data_store():
#     documents = load_documents()
#     chunks = split_text(documents)
#     save_to_chroma(chunks)


# single prompt
# response = llm.invoke("Hello, how are you?")
# print(response)

# multiple prompts
# response = llm.batch(["Hello, how are you?", "Write a poem about AI"])
# print(response)

# receive response in chunks
# response = llm.stream("Write a poem about AI")
# for chunk in response:
    # print(chunk.content, end="", flush=True)


# load text files from a directory
# def load_documents():
#     loader = DirectoryLoader('./data', glob="**/*.md", loader_cls=TextLoader)
#     documents = loader.load()
#     print(len(documents))
#     # print the first 100 characters from the first document
#     # print(documents[0].page_content[:100])
#     return documents

# split loaded text into chunks
# def split_text(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=100,
#         chunk_overlap=50,
#         length_function=len,
#         add_start_index=True,
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(len(chunks))
#     document = chunks[0]
#     print(document.page_content)
#     print(document.metadata)
#     return chunks

# chroma
# def save_to_chroma(chunks):
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH) 

#     db= Chroma.from_documents(
#         chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
#     )
#     db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# if __name__ == "__main__":
#     main()


# evaluator
evaluator = load_evaluator("pairwise_embedding_distance")
x = evaluator.evaluate_string_pairs(prediction="apple", prediction_b="orange")
print(x)