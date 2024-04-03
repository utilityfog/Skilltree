import os
import asyncio
import nbformat
import fitz # PyMuPDF for PDF processing
import pytesseract  # Assuming you choose pytesseract for OCR
from langchain_community.document_transformers.embeddings_redundant_filter import _DocumentWithState

from dotenv import load_dotenv
# pdf, ipynb, txt
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader, TextLoader, NotebookLoader
# Code
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores.pgvector import PGVector
from langchain.vectorstores.deeplake import DeepLake
from langchain.text_splitter import Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from nbconvert import MarkdownExporter
from PIL import Image

from app.embeddings_manager import get_session_vector_store, store_embedding, get_current_session_id
from app.config import EMBEDDING_MODEL, PG_COLLECTION_NAME

client = OpenAI()

load_dotenv()

def custom_embed_entities(client: OpenAI, entities, model="text-embedding-ada-002"):
    """
    Embed a list of documents using OpenAI's API directly, bypassing tiktoken.
    :param documents: A list of strings (documents) to be embedded.
    :param openai_api_key: Your OpenAI API key.
    :param model: The model to use for embeddings.
    :return: A list of embeddings.
    """
    embeddings = []
    for entity in entities:
        try:
            response = client.embeddings.create(input=entity, model=model)
            embeddings.append(response['data'][0]['embedding'])
        except Exception as e:
            print(f"Error embedding document: {e}")
            embeddings.append(None)  # Or handle as appropriate
    return embeddings

class jupyter_notebook_creator:
    def extract_questions(self):
        """Method that utilizes the chat completions api to extract questions that need to be solved from current list of passed files."""
        # This method will be called from server.py and we will first test if it can list all currently passed files correctly.
        
    # def create_notebook_from_questions(questions):
    #     nb = nbformat.v4.new_notebook()
    #     for question in questions:
    #         # Generate markdown cell for the question
    #         question_cell = nbformat.v4.new_markdown_cell(f"### Question:\n{question}")
    #         nb.cells.append(question_cell)
            
    #         # Use chat completions API to get an answer or code snippet
    #         answer = get_chat_completion(question)
            
    #         # Determine if the answer should be a code cell or markdown cell
    #         if is_code_snippet(answer):
    #             answer_cell = nbformat.v4.new_code_cell(answer)
    #         else:
    #             answer_cell = nbformat.v4.new_markdown_cell(answer)
            
    #         nb.cells.append(answer_cell)
        
    #     # Save the new notebook
    #     with open("output_notebook.ipynb", "w", encoding="utf-8") as f:
    #         nbformat.write(nb, f)

class InitialProcessor:
    def process_image(self, file_path):
        # Convert the image file to text using OCR
        text = pytesseract.image_to_string(Image.open(file_path))
        return text
    
    def process_ipynb(self, file_path):
        """Method that reads a jupyter notebook cell by cell and passes each one individually to an LLM API endpoint"""
        with open(file_path, encoding='utf8') as f:
            nb = nbformat.read(f, as_version=4)
        exporter = MarkdownExporter()
        body, _ = exporter.from_notebook_node(nb)
        return body
    
    # def process_pdf(self, file_path):
    #     text = ""
    #     with fitz.open(file_path) as doc:
    #         for page in doc:
    #             text += page.get_text()
    #     return text

    # def process_text_file(self, file_path):
    #     with open(file_path, 'r') as file:
    #         text = file.read()
    #     return text
    
class FileProcessor:
    async def process_documents(self, directory_path, glob_pattern, loader_cls, text_splitter, embeddings, session_vector_store=None):
        """To ensure that your process_documents method accurately processes only PDF files, you can modify the glob_pattern parameter used in the DirectoryLoader to specifically target PDF files. This adjustment will make the method more focused and prevent it from attempting to process files of other types, which might not be suitable for the intended processing pipeline."""
        # pdf_glob_pattern = "**/*.pdf"  # Updated to specifically target PDF files
        print(f"loader_cls: {loader_cls}")
        loader = DirectoryLoader(
            os.path.abspath(directory_path),
            glob=glob_pattern,  # Use the PDF-specific glob pattern
            use_multithreading=True,
            show_progress=True,
            max_concurrency=50,
            loader_cls=loader_cls,
        )
        docs = loader.load()
        # Split documents into meaningful chunks
        chunks = text_splitter.split_documents(docs)
        store, current_embeddings = await PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            chunk_size=100,
            collection_name=PG_COLLECTION_NAME,
            connection_string=os.getenv("POSTGRES_URL"),
            pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
        )
        
        if session_vector_store:
            store, current_embeddings = await session_vector_store.from_documents(
                documents=chunks,
                embedding=embeddings,
                chunk_size=100,
                collection_name=get_current_session_id(),
                connection_string=os.getenv("POSTGRES_URL"),
                pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
            )
        print(f"chunks vectorized: {len(chunks)}")
        return current_embeddings

    async def process_code(self, repo_path, suffixes, language_setting, embeddings, parser_threshold=500, session_vector_store=None):
        print("Began Processing Code")
        # Configure the loader with appropriate settings
        loader = GenericLoader.from_filesystem(
            path=repo_path,
            glob="**/*",
            suffixes=suffixes,
            exclude=[f"**/non-utf8-encoding{suffixes[0]}"],  # Exclude all files that are non-utf8-encoding
            show_progress=True,
            parser=LanguageParser(language=language_setting, parser_threshold=parser_threshold),
        )

        documents = loader.load()

        # Split code into meaningful chunks
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language_setting, chunk_size=100, chunk_overlap=10
        )
        chunks = code_splitter.split_documents(documents)

        store, current_embeddings = await PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            chunk_size=100,
            collection_name=PG_COLLECTION_NAME,
            connection_string=os.getenv("POSTGRES_URL"),
            pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
        )
        if session_vector_store:
            store, current_embeddings = await session_vector_store.from_documents(
                documents=chunks,
                embedding=embeddings,
                chunk_size=100,
                collection_name=get_current_session_id(),
                connection_string=os.getenv("POSTGRES_URL"),
                pre_delete_collection=False,  # Controls whether to clear the collection before adding new docs
            )
        print(f"chunks vectorized: {len(chunks)}")
        return current_embeddings

class FileEmbedder:
    def __init__(self, session_id, pre_delete_collection=True):
        self.session_vector_store = get_session_vector_store(session_id, pre_delete_collection) # pre_delete_collection = True by default
    async def process_and_embed_file(self, unique_id, directory_path, embeddings, glob_pattern=None, suffixes=None, language_setting=None, text_splitter=None):
        # Initialize FileProcessor to access its processing methods
        file_processor = FileProcessor()
        
        # Mapping of file extensions to processing functions and their arguments
        process_map = {
            'pdf': {"func": file_processor.process_documents, "args": {"loader_cls": UnstructuredPDFLoader}},
            'ipynb': {"func": file_processor.process_documents, "args": {"loader_cls": NotebookLoader}},
            'txt': {"func": file_processor.process_documents, "args": {"loader_cls": TextLoader}},
            'code': {"func": file_processor.process_code, "args": {"suffixes": suffixes, "language_setting": language_setting}},
            # Image processing to be implemented later
        }
        
        # Determine the file type from the glob_pattern or suffixes
        file_type = "code" if suffixes else glob_pattern.split(".")[-1]
        
        # Fetch the processing function and its specific arguments
        processing_info = process_map.get(file_type, None)
        
        if processing_info:
            process_func = processing_info["func"]
            process_args = processing_info["args"]

            if file_type in ['pdf', 'ipynb', 'txt']:
                current_embeddings = await process_func(directory_path=directory_path, glob_pattern=glob_pattern, **process_args, text_splitter=text_splitter, embeddings=embeddings, session_vector_store=self.session_vector_store)
            elif file_type == 'code':
                current_embeddings = await process_func(repo_path=directory_path, **process_args, embeddings=embeddings, session_vector_store=self.session_vector_store)
            else:
                # Handling for not yet implemented or unrecognized file types
                return {"error": f"Processing for file type {file_type} is not implemented or unrecognized."}
            
            # Assuming 'current_embeddings' are returned correctly from processing functions
            if current_embeddings:
                print(f"File with file_embedding_key: {unique_id}")
                store_embedding(unique_id, current_embeddings)
        else:
            return {"error": f"No processing function found for file type: {file_type}"}

class RepositoryProcessor:
    def __init__(self, clone_path, deeplake_path, embedding_method, allowed_extensions=None):
        if allowed_extensions is None:
            allowed_extensions = ['.py', '.ipynb', '.md']
        self.clone_path = clone_path
        self.deeplake_path = deeplake_path
        self.allowed_extensions = allowed_extensions
        self.docs = []
        self.texts = []
        self.model_name = EMBEDDING_MODEL
        # "sentence-transformers/all-MiniLM-L6-v2"
        self.model_kwargs = {"model_name": self.model_name}
        self.hf = embedding_method
        # HuggingFaceEmbeddings(**self.model_kwargs)

    def extract_all_files(self):
        for dirpath, dirnames, filenames in os.walk(self.clone_path):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in self.allowed_extensions:
                    try:
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e:
                        print(f"Failed to process file {file}: {e}")

    def chunk_files(self):
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)

    def embed_and_store(self):
        db = DeepLake(dataset_path=self.deeplake_path, embedding_function=self.hf)
        db.add_documents(self.texts)
        print(f"Embedded and stored {len(self.texts)} documents.")

    def process_repository(self):
        self.extract_all_files()
        self.chunk_files()
        self.embed_and_store()
        
    def delete_embeddings(self):
        db = DeepLake(dataset_path=self.deeplake_path, embedding_function=self.hf)
        db.delete(delete_all=True, large_ok=True)
        print(f"Deleted documents in {self.deeplake_path}.")

async def main():
    # Initialize the embeddings and text splitter
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), skip_empty=True, chunk_size=100)
    text_splitter = SemanticChunker(embeddings=embeddings)
    file_processor = FileProcessor()

    # Process PDFs
    # await file_processor.process_documents("./source_docs", "**/*.pdf", UnstructuredPDFLoader, text_splitter, embeddings)

    # Process Code Files (main languages Using the language parser itself)
    # await process_code("/Users/james/Desktop/GitHub/pdf_rag/venv/lib/python3.11/site-packages/langserve", [".py"], Language.PYTHON, embeddings)
    # await process_code("/Users/james/Desktop/GitHub/pdf_rag/venv/lib/python3.11/site-packages/langserve", [".js"], Language.JS, embeddings)
    # await process_code("/Users/james/Desktop/GitHub/pdf_rag/venv/lib/python3.11/site-packages/langserve", [".html"], Language.HTML, embeddings)
    # await process_code("/Users/james/Desktop/GitHub/pdf_rag/venv/lib/python3.11/site-packages/langserve", [".ts"], Language.TS, embeddings)
    
    # await file_processor.process_code("/Users/james/Desktop/GitHub/pdf_rag/frontend", [".py"], Language.PYTHON, embeddings)
    # await file_processor.process_code("/Users/james/Desktop/GitHub/pdf_rag/frontend", [".js"], Language.JS, embeddings)
    
    # await file_processor.process_code("/Users/james/Desktop/GitHub/pdf_rag/app", [".py"], Language.PYTHON, embeddings)
    # await file_processor.process_code("/Users/james/Desktop/GitHub/pdf_rag/app", [".js"], Language.JS, embeddings)
    
    # await file_processor.process_code("/Users/james/Desktop/GitHub/pdf_rag/importer", [".py"], Language.PYTHON, embeddings)
    # await file_processor.process_code("/Users/james/Desktop/GitHub/pdf_rag/importer", [".js"], Language.JS, embeddings)
    
    # await file_processor.process_code("/Users/james/Desktop/GitHub/pdf_rag/venv/lib/python3.11/site-packages/langserve", [".py"], Language.PYTHON, embeddings)
    # await file_processor.process_code("/Users/james/Desktop/GitHub/pdf_rag/venv/lib/python3.11/site-packages/langserve", [".js"], Language.JS, embeddings)
    
    # await file_processor.process_code("./source_code", [".py"], Language.PYTHON, embeddings)
    # await file_processor.process_code("./source_code", [".js"], Language.JS, embeddings)
    # await file_processor.process_code("./source_code", [".ts"], Language.TS, embeddings)

    # Repository Processor for code in multiple languages (after conversion of code to text)
    # repo_processor = RepositoryProcessor(clone_path="/Users/james/Desktop/GitHub/pdf_rag", deeplake_path=os.path.expanduser("~/deeplake_data"), allowed_extensions=[".py", ".js", ".md", ".ts", ".tsx", ".html", ".css"])
    # print(os.path.expanduser("~/deeplake_data"))
    # repo_processor.process_repository()
    # repo_processor.delete_embeddings()

if __name__ == "__main__":
    asyncio.run(main())
