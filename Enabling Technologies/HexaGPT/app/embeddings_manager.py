import os
from typing import Optional
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings

from app.config import EMBEDDING_MODEL

# Initialize the embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=())

# Session ID for current session vector store
CURRENT_SESSION_ID = ""

def set_current_session_id(session_id: str):
    global CURRENT_SESSION_ID
    CURRENT_SESSION_ID = session_id
    print(f"Global session ID set to: {CURRENT_SESSION_ID}")

def get_current_session_id() -> str:
    return CURRENT_SESSION_ID

# Current session vector store
def get_session_vector_store(session_id, pre_delete_collection=True):
    print(f"fucking initializing session vector store via file embedder initialization with session id: {session_id}")
    return PGVector(
        collection_name=session_id,
        connection_string=os.getenv("POSTGRES_URL"),
        embedding_function=embeddings,
        pre_delete_collection=pre_delete_collection
    )
    
def reset_session_vector_store(session_id):
    print(f"fucking resetting session vector store with session id: {session_id}")
    return PGVector(
        collection_name=session_id,
        connection_string=os.getenv("POSTGRES_URL"),
        embedding_function=embeddings,
        pre_delete_collection=True
    ).delete_collection()

# Session EMBEDDINGS_STORAGE
EMBEDDINGS_STORAGE = {}

# Store and get specific embedding given key
def get_embedding_from_manager(unique_id):
    global EMBEDDINGS_STORAGE
    return EMBEDDINGS_STORAGE.get(unique_id)

def store_embedding(unique_id, embedding):
    global EMBEDDINGS_STORAGE
    EMBEDDINGS_STORAGE[unique_id] = embedding

# Get or reset embeddings storage
def get_embeddings_storage():
    return EMBEDDINGS_STORAGE

def reset_embeddings_storage():
    global EMBEDDINGS_STORAGE
    EMBEDDINGS_STORAGE = {}