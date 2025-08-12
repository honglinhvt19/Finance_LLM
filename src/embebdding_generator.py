import os
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from utils import load_config, setup_env

setup_env()
config = load_config()

def load_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    return [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks_data]

def generate_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embedding_model"],
        model_kwargs={"device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"}
    )
    project_root = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(project_root, config["vector_store_dir"])
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    chunks_dir = os.path.join(project_root, config["processed_chunks_dir"])
    for json_file in os.listdir(chunks_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(chunks_dir, json_file)
            chunks = load_chunks(json_path)
            vectordb = generate_embeddings(chunks)
            print(f"Generated embeddings for {json_file} using Vietnamese model")