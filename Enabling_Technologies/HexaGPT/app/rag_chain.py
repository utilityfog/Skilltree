import json
import re
import os
import traceback
import requests

from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
from abc import ABC
from functools import partial
from dotenv import load_dotenv

from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
    LongContextReorder
)

# from langchain_core.vectorstores import VectorStoreRetriever, Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import Runnable
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.vectorstores.deeplake import DeepLake

from langchain.retrievers import (
    ContextualCompressionRetriever, 
    EnsembleRetriever, 
    MergerRetriever
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from pypdf import PdfReader

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.embeddings_manager import get_current_session_id, get_embedding_from_manager, get_embeddings_storage, get_persistent_vector_store, get_session_vector_store, EMBEDDINGS_STORAGE, reset_embeddings_storage, reset_session_vector_store, set_current_session_id
from app.config import PG_COLLECTION_NAME, EMBEDDING_MODEL
from app.JupyterAutocomplete import JupyterSolver

load_dotenv()

# Google Search Environmental Variables
google_api_key = os.environ.get('GOOGLE_API_KEY')
google_cse_id = os.environ.get('GOOGLE_CUSTOM_ENGINE_ID')

def augment_context_with_file_embeddings(context: str, file_embedding_keys: Optional[Optional[str]]) -> str:
    augment_text = ""
    if file_embedding_keys:
        print(f"file_embedding_keys current: {file_embedding_keys}")
        for file_embedding_key in file_embedding_keys:
            file_embedding = get_embedding_from_manager(file_embedding_key)
            if file_embedding:
                augment_text += f"\nFrom {file_embedding_key}: {file_embedding}"
            else:
                print("Why is fetched embedding from current embedding_store none?")
        augment_text = f"\n augment: {augment_text}, Context: " + f"{context}"
        # trimmed_print("final augmented context", augment_text, 10000) # This does print something
        reset_embeddings_storage() # Reset after each augmented generation
        reset_session_vector_store(get_current_session_id()) # Reset after each augmented generation
        set_current_session_id("") # Reset after each augmented generation
    return augment_text

# Initialize the embeddings
embedding_method = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), show_progress_bar=True, chunk_size=100, skip_empty=True)

persistent_vector_store = get_persistent_vector_store()

# PGVector(
#     collection_name=PG_COLLECTION_NAME,
#     connection_string=os.getenv("POSTGRES_URL"),
#     embedding_function=embedding_method
# )

# Initialize DeepLake dataset
# deeplake_path = os.path.expanduser("~/deeplake_data")
# deeplake_db = DeepLake(dataset_path=deeplake_path, embedding_function=embeddings, read_only=True)

# initialize the ensemble retriever
# ensemble_retriever = EnsembleRetriever(retrievers=[vector_store.as_retriever(), deeplake_db.as_retriever()], weights=[0.5, 0.5])

# Default PGVectorStore retriever
pgvector_default_retriever = persistent_vector_store.as_retriever(
    search_type="mmr",
    # search_kwargs={"k": 6, "include_metadata": True}
    search_kwargs={'k': 6, 'lambda_mult': 0}
)
# With Score PGVectorStore retriever
pgvector_score_retriever = persistent_vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.01}
)

# deeplake_retriever = deeplake_db.as_retriever(
#     search_type="mmr", search_kwargs={"k": 9, "include_metadata": True}
# )

# The Lord of the Retrievers will hold the output of both retrievers and can be used as any other
# retriever on different types of chains.
lotr = MergerRetriever(retrievers=[pgvector_default_retriever])

# This filter will divide the documents vectors into clusters or "centers" of meaning.
# Then it will pick the closest document to that center for the final results.
# By default the result document will be ordered/grouped by clusters.
filter_ordered_cluster = EmbeddingsClusteringFilter(
    embeddings=embedding_method,
    num_clusters=3,
    num_closest=1,
)

# If you want the final document to be ordered by the original retriever scores
# you need to add the "sorted" parameter.
filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=embedding_method,
    num_clusters=2,
    num_closest=1,
    # random_state=None,
    sorted=True
    # remove_duplicates=True
)

# We can remove redundant results from both retrievers using yet another embedding.
# Using multiples embeddings in diff steps could help reduce biases.
embeddings_filter = EmbeddingsRedundantFilter(embeddings=embedding_method, similarity_threshold=0.97)
pipeline = DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever, embeddings_filter])
# Default
ordered_compression_default_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=pgvector_default_retriever
)
# Score retriever
ordered_compression_score_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=pgvector_score_retriever
)

# Retrieved context:
# {context}
CONVERSATION_TEMPLATE = """
Answer given the following:

Uploaded file augment: {augment}

Question from: {question}
"""

PREPROCESSING_TEMPLATE = """
Extract a full list of problems that need to be solved from the augment:

Uploaded file augment: {augment}

Retrieved context:
{context}

Question from: {question}

Note:
- If a single Jupyter notebook contains the problems that need to be solved, extract all relevant problems from the notebook.
- If dealing with a PDF with a comprehensive list of problems AND a PDF that lists the problems that need to be solved is NOT uploaded, extract all relevant problems from the PDF with a comprehensive list of problems.
- If a PDF that points to another document for problems is uploaded, identify and extract ONLY the relevant problems from the referred document.

Each problem requires a label for both the problem number it is part of and sub problem it is part of.
"""

RESEARCH_TEMPLATE = """
Explain academic papers given the following:

Retrieved context:
{context}

Question from: {question}
"""

def template_processor(embeddings_storage, template):
    final_template = template
    # for key in embeddings_storage:
    #     print(f"appending key {key} to template")
    #     final_template += f"""\nInformation from file with {key}"""
    print(final_template)
    # print(traceback.print_stack())
    return final_template

# Refer to the uploaded file information: {file_embedding_key}
#ANSWER_PROMPT = ChatPromptTemplate.from_template(template_processor(embeddings_storage, template))

llm_conversation = ChatOpenAI(temperature=0.7, model='gpt-4o', max_tokens=4096, streaming=True)

llm_jupyter_preprocessor = ChatOpenAI(temperature=0.7, model='gpt-4o', max_tokens=4096, streaming=True, tools=JupyterSolver.solver_tools, tool_choice={"type": "function", "function": {"name": "solve_problems"}}) # Any additional argument we pass here will become model_kwargs

print(f"max_tokens: {llm_conversation.max_tokens}")
print(f"model_name: {llm_conversation.model_name}")

class RagInput(TypedDict, total=False):
    question: str
    fetchContext: Optional[bool] # Optional bool for whether or not to fetch context
    file_embedding_keys: Optional[List[str]] # Optional list of strings
    
class JupyterRagInput(TypedDict, total=False):
    question: str

# For JSON Runnable Parallel
# RunnableParallel(
#     context=(itemgetter("question") | vector_store.as_retriever()),
#     question=itemgetter("question")
# ) |
# RunnableParallel(
#     answer=(ANSWER_PROMPT | llm),
#     docs=itemgetter("context")
# )

# For String output
# {
#     "context": (itemgetter("question") | vector_store.as_retriever()),
#     "question": itemgetter("question")
# } | ANSWER_PROMPT | llm | StrOutputParser()

# class print_stack_class:
#     def print_stack(self, **kwargs):
#         print(traceback.print_stack())
        
# print_stack_class_instance = print_stack_class()

def trimmed_print(label, data, max_length=50):
    data_str = str(data)
    trimmed_data = (data_str[:max_length] + '...') if len(data_str) > max_length else data_str
    print(f"{label}: {trimmed_data}")

def get_augment_and_trace(dictionary):
    """Method to augment response with uploaded file embeddings"""
    keys = dictionary['file_embedding_keys']
    session_id = get_current_session_id()
    print(f"session_id before augment: {session_id}")
    print(f"keys: {keys}")
    summarized_embeddings = []
    augmented_embeddings = []
    if session_id:
        print("Does this run when not session_id")
        session_vector_store = get_session_vector_store(session_id, pre_delete_collection = False)
        persistent_vector_store = get_persistent_vector_store()
        num_keys = 0
        for key in keys: # each key here is analogous to each unique_id
            print(f"key_{num_keys}: {key}")
            num_keys += 1
            embeddings_for_file_with_key = get_embedding_from_manager(key) # embeddings = List[List[float]]
            # trimmed_print("embeddings_for_file_with_key", embeddings_for_file_with_key, 10000)
            # trimmed_print("fucking embeddings: ", embeddings, 10000)
            if embeddings_for_file_with_key:
                for embedding in embeddings_for_file_with_key:
                    # trimmed_print("fucking_embedding: ", embedding, 10000) // This is proven to be different for different keys
                    summarized_embedding = session_vector_store.similarity_search_by_vector(embedding=embedding, k=1)
                    # augmented_embedding = persistent_vector_store.similarity_search_with_relevance_scores( # return type == List[Tuple(Document, float)]
                    #     embedding=embedding,
                    #     k=1,
                    #     score_threshold=0.85
                    # )
                    # Also fetch additional context for each embedding from existing persistent vector store
                    # trimmed_print("fucking summarized embedding for current file: ", summarized_embedding, 10000)
                    summarized_embeddings.append(summarized_embedding)
                    # augmented_embeddings.append(augmented_embedding)
    print(f"summarized embeddings length: {len(str(summarized_embeddings))}")
    # trimmed_print("uploaded file to augment", summarized_embeddings, 10000)
    # summarized_embeddings.extend(augmented_embeddings)
    trimmed_print("fucking final context", summarized_embeddings, 10000)
    return summarized_embeddings

fetch_context = True
def get_question(inputs):
    """Method to get question"""
    print(f"inputs: {inputs.keys()}")
    question = inputs['question']
    global fetch_context
    fetch_context = inputs['fetchContext']
    print(f"question: {question}")
    return question
    
def retrieve_context_from_question(question):
    """Method to retrieve context from question"""
    if fetch_context:
        return ordered_compression_default_retriever
    else:
        return
    

def get_context_and_trace(dictionary, key):
    # Print the stack trace when this function is called
    # print(f"fucking docs context stacktrace: {traceback.print_stack()}")
    context = dictionary[key]
    # for document in context:
        # trimmed_print("fucking docs context", document, 1000)
    print(f"context length: {len(str(context))}")
    # trimmed_print("fucking context", context, 10000)
    return context

def redirect_test(dictionary, key):
    # Print the stack trace when this function is called
    # print(f"fucking docs context stacktrace: {traceback.print_stack()}")
    question = dictionary[key]
    # for document in context:
        # trimmed_print("fucking docs context", document, 1000)
    print(f"redirected to solver!: {len(str(question))}")
    return question

# def process_response(dictionary, key):
#     """Method that processes initial LLM response into a format that can be received by JupyterSolver.process_files_and_generate_response"""
#     response = dictionary[key]
#     print(f"process_response called: {response}")
    
#     # Assuming 'response' is an instance of AIMessageChunk and 'additional_kwargs' is an attribute of it
    # if hasattr(response, 'additional_kwargs'):
    #     tool_calls = response.additional_kwargs.get('tool_calls', [])
    # else:
    #     tool_calls = []
    # print(f"tool_calls: {tool_calls}")
    
#     extracted_data = {}
#     for tool_call in tool_calls:
#         if 'function' in tool_call and tool_call['function'].get('name') == "process_files_and_generate_response":
#             arguments_json = tool_call['function'].get('arguments', '{}')
#             arguments_dict = json.loads(arguments_json)
#             # Directly extracting 'problem_file_types' and 'problem_files'
#             problem_file_types = arguments_dict.get("dictionary", {}).get("problem_file_types", [])
#             problem_files = arguments_dict.get("dictionary", {}).get("problem_files", [])
#             extracted_data = {
#                 "problem_file_types": problem_file_types,
#                 "problem_files": problem_files
#             }
#             print(f"Extracted Data: {extracted_data}")
#             break
#     else:
#         extracted_data = "Error: function process_files_and_generate_response does not exist"
    
#     return extracted_data # Should be Dict that has 2 keys: problem_file_types, problem_files; for problem_file_types the value is a List of file types in .extension form, for problem_files, the value is a List of actual Document objects that contain the problems that need to be solved.

# def process_response(dictionary, key):
#     """Extract questions, problem file path, and pointer file path from the LLM response."""
#     response = dictionary[key]
#     print(f"process_response called: {response}")

#     # Initialize default values
#     problems = []
#     problem_file_path = ""
#     pointer_file_path = ""

#     if 'tool_calls' in response:
#         for tool_call in response['tool_calls']:
#             if tool_call['function']['name'] == "solve_problems":
#                 arguments_dict = tool_call['function']['arguments']
#                 problems = arguments_dict.get("problems", [])
#                 problem_file_path = arguments_dict.get("problem_file_path", "")
#                 pointer_file_path = arguments_dict.get("pointer_file_path", "")
#                 print(f"Extracted: Problems - {problems}, Problem File Path - {problem_file_path}, Pointer File Path - {pointer_file_path}")
#                 break
#         else:
#             print("Error: function solve_problems does not exist or no questions extracted.")
#     else:
#         print("No tool calls found in response.")

#     return problems, problem_file_path, pointer_file_path

def preprocess_json_string(arguments_json):
    """Preprocess JSON string to escape problematic characters safely."""
    # Escaping backslashes first to avoid double escaping
    arguments_json = arguments_json.replace('\\', '\\\\')
    
    # Escaping double quotes within the string, ensuring not to escape legitimate JSON structure
    # This simplistic approach might not be foolproof for all edge cases
    arguments_json = re.sub(r'(?<!\\)"', '\\"', arguments_json)
    
    return arguments_json

def process_response(dictionary, key):
    """Extract questions, problem file path, and pointer file path from the LLM response."""
    response = dictionary[key]
    problems = []
    problem_file_path = ""
    pointer_file_path = ""
    print(f"response: {response}")
        
    if hasattr(response, 'additional_kwargs'):
        tool_calls = response.additional_kwargs.get('tool_calls', [])
    else:
        tool_calls = []
    print(f"tool_calls: {tool_calls}")
    
    for tool_call in tool_calls:
        if tool_call['function']['name'] == "solve_problems":
            arguments_json = tool_call['function'].get('arguments', '{}')
            print(f"arguments_json: {arguments_json}")
            # Preprocessing the JSON string to escape problematic characters
            preprocessed_json = preprocess_json_string(arguments_json)
            # try:
            #     arguments_dict = json.loads(arguments_json)
            # except json.JSONDecodeError as e:
            #     print(f"Error decoding JSON: {e}")
            #     continue  # or handle error as appropriate
            
            problems = arguments_json.get("problems", [])
            problem_file_path = arguments_json.get("problem_file_path", "")
            pointer_file_path = arguments_json.get("pointer_file_path", "")
            print(f"Extracted: Problems - {problems}, Problem File Path - {problem_file_path}, Pointer File Path - {pointer_file_path}")
            break
        else:
            print("Error: function solve_problems does not exist or no questions extracted.")

    return problems, problem_file_path, pointer_file_path

async def fetch_context_document_with_score(inputs, retriever=ordered_compression_score_retriever):
    search_query = inputs['question']
    results = await retriever.base_retriever.vectorstore.asimilarity_search_with_relevance_scores( # return type == List[Tuple(Document, float)]
        search_query, 
        9, 
        score_threshold=0.95
    )
    run_searches = False
    if not results:
        run_searches = True
        return results, run_searches
    
    for result in results:
        print(f"similarity: {result[1]}")
    
    trimmed_print("trimmed similarity search with score: ", results, 1000)
    return results, run_searches

# Site restriction guiding prompt to search term
sites = ["site: https://arxiv.org"]
google_guidance_prompts = [f"{site} " for site in sites]

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build(serviceName="customsearch", version="v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

# Function to run searches
def run_google_searches(inputs, api_key=google_api_key, cse_id=google_cse_id, guidance_prompts=google_guidance_prompts):
    """Method that pipes context fetcher and runs google searches if context does not include the paper a user mentioned in its top 3 papers, determined by the received maximum similarity 
    between the piped search_query and the chunk of any pdf within current context, which should be >=98%
    """
    search_query = inputs['question']
    # should_run_searches = inputs['run_searches']
    
    #if should_run_searches: Assume that we should always run searches
    for guidance_prompt in guidance_prompts:
        full_search_term = f"{guidance_prompt} {search_query}"
        try:
            results = google_search(full_search_term, api_key, cse_id, num=8)
            print(f"results: {results}")
            return results
        except HttpError as error:
            print(f"An error occurred: {error}")
        print("\n----------------------------------------\n")
        
async def run_google_searches_and_embed_pdfs(inputs, api_key=google_api_key, cse_id=google_cse_id, guidance_prompts=google_guidance_prompts, embedding_method=embedding_method):
    """Extension of run_google_searches to download PDFs and embed them."""
    search_query = inputs['question']
    for guidance_prompt in guidance_prompts:
        full_search_term = f"{guidance_prompt} {search_query}"
        try:
            results = google_search(full_search_term, api_key, cse_id, num=8)
            pdf_urls = [result['link'] for result in results if result['link'].endswith('.pdf')]
            embeddings = []
            for url in pdf_urls:
                response = requests.get(url)
                with open('temp.pdf', 'wb') as f:
                    f.write(response.content)
                # Process document given directory
                reader = PdfReader('temp.pdf')
                full_text = '\n'.join(page.extract_text() for page in reader.pages if page.extract_text())
                embeddings.append(embedding_method.embed(full_text))
            return embeddings
        except HttpError as error:
            print(f"An error occurred: {error}")
        print("\n----------------------------------------\n")
            
# class CustomVectorStoreRetriever(VectorStoreRetriever):
#     """
#     A custom retriever that extends the basic VectorStoreRetriever to include
#     similarity scores and potentially the full content of documents in its results.
#     """
    
#     def __init__(self, store):
#         # super().__init__()
#         self = store.as_retriever(
#             search_type="similarity", search_kwargs={"k": 9, "include_metadata": True}
#         )

#     def _get_relevant_documents(self, query: str, *, run_manager) -> List[Tuple[Document, float]]:
#         """
#         Override to fetch documents along with their similarity scores.
#         """
#         if self.search_type == "similarity":
#             # Fetch documents and their similarity scores
#             docs_and_scores = self.max_marginal_relevance_search_with_score(query, **self.search_kwargs)
#             return docs_and_scores
#         else:
#             raise ValueError(f"Unsupported search type: {self.search_type}")

#     def get_documents_with_scores(self, query: str) -> List[Tuple[Document, float]]:
#         """
#         A public method to retrieve documents along with their similarity scores.
#         """
#         return self._get_relevant_documents(query, run_manager=None)

# Correctly setting up partial functions
custom_question_getter = partial(get_question)
question_context_retriever_runnable = RunnableLambda(lambda question_runnable: retrieve_context_from_question(question_runnable))
custom_context_getter = partial(get_context_and_trace, key="context") # change to context
custom_augment_getter = partial(get_augment_and_trace)
redirect_test_runnable = partial(redirect_test, key="question")
process_response_runnable = partial(process_response, key='response')

jupyter_solver = JupyterSolver()

# preprocess_chain is a sequential(not parallel) chain that does the following:
# 1. Passes the context, augment, and file_embedding_keys to the chat completions api and receives a json that identifies the function that must be called with arguments. This function is process_files_and_generate_response of the JupyterSolver Class
# 2. Calls process_files_and_generate_response with proper arguments that were returned by the chat completions api.
# 3. Once the document(s) that hold the questions are identified through the first chain that uses the chat completions api, solve_chain will pipe that to another chain that this time uses the chat completions api to
# extract a list of problem strings from the document(s) that hold the questions.
# 4. Generate a jupyter notebook where each cell correspond to each extracted problem from the document augment.
preprocess_chain = (
    # This chain needs to be modified so that the final return value is the json that specifies the function that needs to be called with arguments.
    (RunnableParallel(
        context=(itemgetter("question") | ordered_compression_default_retriever),
        augment=custom_augment_getter,
        question=redirect_test_runnable,
        file_embedding_keys=itemgetter("file_embedding_keys")
    ) |
    RunnableParallel(
        response=(ChatPromptTemplate.from_template(template=PREPROCESSING_TEMPLATE) | llm_jupyter_preprocessor),
        docs=custom_context_getter,
        augment=lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
        augments=itemgetter("augment")
    )) |
    process_response_runnable |
    RunnableParallel(
        problem_extraction=lambda inputs: jupyter_solver.solve_problems(problems=inputs[0], problem_file_path=inputs[1], pointer_file_path=inputs[2])
    )
).with_types(input_type=RagInput)

fetch_context = partial(fetch_context_document_with_score)
run_searches_and_embed_results = partial(run_google_searches_and_embed_pdfs)

# customVectorStoreRetriever = CustomVectorStoreRetriever(ordered_compression_retriever) # base retreiver = PGVectorStoreRetriever

# Chain for returning to the user three most relevant academic paper given their query and explaining what they are about
research_chain = (
    RunnableParallel(
        # subchain that normally returns context but returns top 3 pdfs if they exist
        context=run_searches_and_embed_results,
        augment=custom_augment_getter, # type of augment is List[List[Document]]; Can be empty if no files are uploaded
        question=itemgetter("question"),
        file_embedding_keys=itemgetter("file_embedding_keys")
    ) |
    RunnableParallel(
        response=(ChatPromptTemplate.from_template(template=RESEARCH_TEMPLATE) | llm_conversation),
        docs=custom_context_getter,
        augment=lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
        augments=itemgetter("augment")
    )
).with_types(input_type=RagInput)

# Chain for solving passed Jupyter Notebook
solve_chain = (
    RunnableParallel (
        current_problem=(itemgetter("question") | ordered_compression_default_retriever), # Return type of current_cell_context=?
        persistent_cell_context=itemgetter("problem"), # Call the method similar to custom_augment_getter to fetch each List[List[Document]] from the vectorstore
        previous_cells_context=itemgetter("problem"), # Since return type of augment was List[List[Document]], make sure previous_cells_context is the same type when passed from JupyterSolver.send_for_completion
    ) |
    RunnableParallel (
        # Fix.
        response=(ChatPromptTemplate.from_template(template=CONVERSATION_TEMPLATE) | llm_conversation),
        docs=custom_context_getter,
        augment=lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
        augments=itemgetter("augment")
    )
).with_types(input_type=JupyterRagInput)

# Default Chat Conversation Chain
final_chain = (
    RunnableParallel (
        # Type of itemgetter("question") == string
        context=(custom_question_getter | question_context_retriever_runnable),
        augment = custom_augment_getter, # type of augment is List[List[Document]]
        question=itemgetter("question"),
        file_embedding_keys=itemgetter("file_embedding_keys") # Optional
    ) |
    RunnableParallel (
        answer = (ChatPromptTemplate.from_template(template=CONVERSATION_TEMPLATE) | llm_conversation),
        docs = custom_context_getter,
        augment = lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
        augments = itemgetter("augment")
    )
).with_types(input_type=RagInput)
