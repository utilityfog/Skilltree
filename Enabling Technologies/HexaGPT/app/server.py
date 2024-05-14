from typing import List
import uuid
import secrets
from nbclient import NotebookClient
import nbformat

from fastapi import FastAPI, File, Header, UploadFile, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
from langchain.text_splitter import Language
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents.base import Document

from importer.load_and_process import FileEmbedder
from app.config import EMBEDDING_MODEL
from app.embeddings_manager import get_embedding_from_manager, get_session_vector_store, reset_embeddings_storage, reset_session_vector_store, set_current_session_id, store_embedding
from app.rag_chain import final_chain, solve_chain, preprocess_chain, research_chain
from app.JupyterAutocomplete import JupyterSolver
# from app.config import EMBEDDING_MODEL, PG_COLLECTION_NAME

app = FastAPI()

# Mapping from file extensions to Language enumeration values
language_map = {
    'py': Language.PYTHON,
    'js': Language.JS,
    'ts': Language.TS,
    'html': Language.HTML,
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/rag/static", StaticFiles(directory="./source_docs"), name="static")

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Upload file endpoint
@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...), session_id: str = Header(default_factory=secrets.token_urlsafe)):
    set_current_session_id(session_id)  # Update the global session ID
    upload_folder = Path("./uploaded_files")
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize the embeddings and text splitter
    embedding_method = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), show_progress_bar=True, skip_empty=True, chunk_size=100)
    text_splitter = SemanticChunker(embeddings=embedding_method)
    
    # Use session_id to handle embeddings in a session-specific manner.
    file_embedder = FileEmbedder(session_id)
    
    file_responses = []  # Initialize an empty list to store responses
    for file in files:
        unique_id = str(uuid.uuid4())
        save_path = upload_folder / f"{file.filename}"

        with open(save_path, "wb") as out_file:
            out_file.write(await file.read())
            
            # Determine file type and set parameters for processing
            file_extension = file.filename.split('.')[-1].lower()
            print(f"uploaded file_extension: {file_extension}")
            try:
                if file_extension in ['pdf', 'ipynb', 'txt', 'csv']:
                    glob_pattern = f"**/*.{file_extension}"
                    await file_embedder.process_and_embed_file(unique_id=unique_id, directory_path=str(upload_folder), embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter)
                elif file_extension in ['py', 'js', 'ts', 'html']:
                    suffixes = ["."+file_extension] # [".py"]
                    print(suffixes)
                    await file_embedder.process_and_embed_file(unique_id=unique_id, directory_path=str(upload_folder), embeddings=embedding_method, suffixes=suffixes, language_setting=language_map.get(file_extension, None))
                file_responses.append({"filename": file.filename, "unique_id": unique_id})
            except Exception as e:
                print(f"Error processing file {file.filename}: {e}")
            finally:
                # Delete the file after processing it to avoid redundancy
                save_path.unlink(missing_ok=True)  # Use missing_ok=True to ignore the error if the file doesn't exist
            # print(f"file added to: {file_responses}")
            
    if not file_responses:
        raise HTTPException(status_code=400, detail="No files were uploaded.") 
    
    return {"files": file_responses}

@app.get("/embeddings/{unique_id}")
async def get_embedding(unique_id: str):
    # Retrieve the stored embedding
    embedding = get_embedding_from_manager(unique_id)
    if embedding is None:
        print("Why is fetched embedding from current embedding_store none?")
        raise HTTPException(status_code=404, detail="Embedding not found")
    return embedding

@app.get("/files/{file_name}")
async def get_file(file_name: str):
    file_path = f"./uploaded_files/{file_name}"
    return FileResponse(path=file_path, filename=file_name, media_type='application/octet-stream')

@app.post("/process")
async def solve(request: Request):
    """
    Calls a chain sequence that generates a jupyter notebook for a given input.
    """
    
    # Redirect the current request to the newly added solve_chain route
    response = RedirectResponse("/preprocess/stream")

    # Return the response received from solve_chain
    return response

@app.post("/solve_jupyter")
async def solve_jupyter(files: List[UploadFile] = File(...), session_id: str = Header(default_factory=secrets.token_urlsafe)):
    set_current_session_id(session_id)  # Update the global session ID
    upload_folder = Path("./uploaded_files")
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize the embeddings and text splitter
    embedding_method = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), show_progress_bar=True, skip_empty=True, chunk_size=100)
    text_splitter = SemanticChunker(embeddings=embedding_method)
    
    # Use session_id to handle embeddings in a session-specific manner.
    file_embedder = FileEmbedder(session_id) # This step already initializes a session PG Vector Store
    
    # Initialize instance of JupyterSolver class
    jupyter_solver = JupyterSolver()
    
    file_responses = []  # Initialize an empty list to store responses
    for file in files:
        # Check the file extension instead of content type
        if not file.filename.endswith('.ipynb'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a Jupyter notebook.")
        
        unique_id = str(uuid.uuid4())
        save_path = upload_folder / f"{file.filename}"
        
        content = await file.read()  # Read the file content once
        
        with open(save_path, "wb") as out_file:
            out_file.write(content)  # Write the content to a file
            
        nb = nbformat.reads(content.decode('utf-8'), as_version=4)
        num_cells = len(nb.cells)
        print(f"Notebook received with {num_cells} cells.")
            
        # Determine file type and set parameters for processing
        file_extension = file.filename.split('.')[-1].lower()
        print(f"uploaded file_extension: {file_extension}")
        
        try:
            if file_extension in ['pdf', 'ipynb', 'txt']:
                glob_pattern = f"**/*.{file_extension}"
                await file_embedder.process_and_embed_file(unique_id=unique_id, directory_path=str(upload_folder), embeddings=embedding_method, glob_pattern=glob_pattern, text_splitter=text_splitter)
            elif file_extension in ['py', 'js', 'ts', 'html']:
                suffixes = ["."+file_extension] # [".py"]
                print(suffixes)
                await file_embedder.process_and_embed_file(unique_id=unique_id, directory_path=str(upload_folder), embeddings=embedding_method, suffixes=suffixes, language_setting=language_map.get(file_extension, None))
            file_responses.append({"filename": file.filename, "unique_id": unique_id})
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
        finally:
            # Delete the file after processing it to avoid redundancy
            save_path.unlink(missing_ok=True)  # Use missing_ok=True to ignore the error if the file doesn't exist
                            
        # Retrieve the vector embeddings for the current notebook
        embedded_current_notebook = get_embedding_from_manager(unique_id)
        
        # Get current session vector store
        session_vector_store = get_session_vector_store(session_id, pre_delete_collection = False)
        
        current_notebook_documents: List[List[Document]] = jupyter_solver.convert_embeddings_to_documents(embedded_current_notebook, session_vector_store)
        
        # Call autocomplete_and_execute_cells, which iterates through each cell of a jupyter notebook and uses the chat completions create api endpoint to generate a response.
        await jupyter_solver.autocomplete_and_execute_cells(nb, current_notebook_documents, session_vector_store, embedding_method, text_splitter) # This method iterates cells of a notebook. This means the List[Document] argument should reset persistence as we iterate from uploaded notebook to notebook.
            
    if not file_responses:
        raise HTTPException(status_code=400, detail="No files were uploaded.") 
    
    return {"files": file_responses}

add_routes(app, final_chain, path="/rag") # Main

add_routes(app, preprocess_chain, path="/preprocess")

add_routes(app, solve_chain, path="/jupyter")

add_routes(app, research_chain, path="/research")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
