import json
import re
import os
import traceback

from operator import itemgetter
from typing import Dict, List, Optional, TypedDict, Union
from abc import ABC
from functools import partial

from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores.deeplake import DeepLake
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables.base import Runnable
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
    LongContextReorder
)
from langchain.retrievers import (
    ContextualCompressionRetriever, 
    EnsembleRetriever, 
    MergerRetriever
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

from app.embeddings_manager import get_current_session_id, get_embedding_from_manager, get_embeddings_storage, get_session_vector_store, EMBEDDINGS_STORAGE, reset_embeddings_storage, reset_session_vector_store, set_current_session_id
from app.config import PG_COLLECTION_NAME, EMBEDDING_MODEL
from app.JupyterAutocomplete import JupyterSolver

load_dotenv()

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
        augment_text = f"augment: {augment_text}, Context: " + f"{context}"
        trimmed_print("final augmented context", augment_text, 10000) # This does print something
        reset_embeddings_storage() # Reset after each augmented generation
        reset_session_vector_store(get_current_session_id()) # Reset after each augmented generation
        set_current_session_id("") # Reset after each augmented generation
    return augment_text

# Initialize the embeddings
embedding_method = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=(), show_progress_bar=True, chunk_size=100, skip_empty=True)

vector_store = PGVector(
    collection_name=PG_COLLECTION_NAME,
    connection_string=os.getenv("POSTGRES_URL"),
    embedding_function=embedding_method
)

# Initialize DeepLake dataset
# deeplake_path = os.path.expanduser("~/deeplake_data")
# deeplake_db = DeepLake(dataset_path=deeplake_path, embedding_function=embeddings, read_only=True)

# initialize the ensemble retriever
# ensemble_retriever = EnsembleRetriever(retrievers=[vector_store.as_retriever(), deeplake_db.as_retriever()], weights=[0.5, 0.5])

# Define 2 diff retrievers with 2 diff embeddings and diff search type.
pgvector_retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 9, "include_metadata": True}
)
# deeplake_retriever = deeplake_db.as_retriever(
#     search_type="mmr", search_kwargs={"k": 9, "include_metadata": True}
# )

# The Lord of the Retrievers will hold the output of both retrievers and can be used as any other
# retriever on different types of chains.
lotr = MergerRetriever(retrievers=[pgvector_retriever])

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
    num_clusters=3,
    num_closest=1,
    random_state=0,
    sorted=True
    # remove_duplicates=True
)

# We can remove redundant results from both retrievers using yet another embedding.
# Using multiples embeddings in diff steps could help reduce biases.
embeddings_filter = EmbeddingsRedundantFilter(embeddings=embedding_method, similarity_threshold=0.95)

pipeline = DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever, embeddings_filter])
ordered_compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=pgvector_retriever
)

CONVERSATION_TEMPLATE = """
Answer given the following:

Uploaded file augment: {augment}

Retrieved context:
{context}

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

llm_conversation = ChatOpenAI(temperature=0.2, model='gpt-4-0125-preview', max_tokens=3500, streaming=True)

llm_jupyter_preprocessor = ChatOpenAI(temperature=0.2, model='gpt-4-0125-preview', max_tokens=3500, streaming=True, tools=JupyterSolver.solver_tools, tool_choice={"type": "function", "function": {"name": "solve_problems"}}) # Any additional argument we pass here will become model_kwargs

print(f"max_tokens: {llm_conversation.max_tokens}")
print(f"model_name: {llm_conversation.model_name}")

class RagInput(TypedDict, total=False):
    question: str
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
    if session_id:
        print("Does this run when not session_id")
        session_vector_store = get_session_vector_store(session_id, pre_delete_collection = False)
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
                    trimmed_print("fucking summarized embedding for current file: ", summarized_embedding, 10000)
                    summarized_embeddings.append(summarized_embedding)
    print(f"summarized embeddings length: {len(str(summarized_embeddings))}")
    trimmed_print("fucking augment", summarized_embeddings, 10000)
    return summarized_embeddings

def get_context_and_trace(dictionary, key):
    # Print the stack trace when this function is called
    # print(f"fucking docs context stacktrace: {traceback.print_stack()}")
    context = dictionary[key]
    # for document in context:
        # trimmed_print("fucking docs context", document, 1000)
    print(f"context length: {len(str(context))}")
    trimmed_print("fucking context", context, 10000)
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

# Correctly setting up partial functions
custom_context_getter = partial(get_context_and_trace, key="context")
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
        context=(itemgetter("question") | ordered_compression_retriever),
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

# Let's instead modify solve_chain for requests to make our lives easier.
solve_chain = (
    RunnableParallel(
        current_cell_context=(itemgetter("problem") | ordered_compression_retriever), # Return type of current_cell_context=?
        persistent_cell_context=(itemgetter("persistent_context")), # Since return type of augment was List[List[Document]], make sure persistent_cell_context is the same type when passed from JupyterSolver.send_for_completion
        previous_cells_context=(itemgetter("previous_context")), # Since return type of augment was List[List[Document]], make sure previous_cells_context is the same type when passed from JupyterSolver.send_for_completion
        problem=itemgetter("problem"),
    ) |
    RunnableParallel(
        # Fix.
        response=(ChatPromptTemplate.from_template(template=CONVERSATION_TEMPLATE) | llm_conversation),
        docs=custom_context_getter,
        augment=lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
        augments=itemgetter("augment")
    )
).with_types(input_type=JupyterRagInput)

final_chain = (
    RunnableParallel(
        context=(itemgetter("question") | ordered_compression_retriever),
        augment = custom_augment_getter,
        question=itemgetter("question"),
        file_embedding_keys=itemgetter("file_embedding_keys")
    ) |
    RunnableParallel(
        answer = (ChatPromptTemplate.from_template(template=CONVERSATION_TEMPLATE) | llm_conversation),
        docs = custom_context_getter,
        augment = lambda inputs: augment_context_with_file_embeddings(inputs["context"], inputs["file_embedding_keys"]),
        augments = itemgetter("augment")
    )
).with_types(input_type=RagInput)
