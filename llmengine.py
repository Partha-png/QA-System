# llm_engine.py

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.settings import Settings
import google.generativeai as genai

# Load environment variables
load_dotenv()
Google_Api_key = os.getenv('Google_Api_key')
genai.configure(api_key=Google_Api_key)

# Initialize models once
gm = GeminiEmbedding(model_name="models/gemini-embedding-exp-03-07")
model = Gemini(models='gemini-1.5-pro', api_key=Google_Api_key)

Settings.llm = model
Settings.embed_model = gm
Settings.chunk_size = 800
Settings.chunk_overlap = 20

def process_txt(file_path):
    """Load documents and build the vector index"""
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index
def ask_question(index, question):
    """Query the index"""
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response.response

