import uuid
from fastapi import Path, Request
from fastapi.responses import RedirectResponse
import httpx
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
import nbformat

from typing import Dict, List, Optional, Tuple
from nbclient import NotebookClient

from app.embeddings_manager import get_current_session_id, get_embedding_from_manager
from importer.load_and_process import FileEmbedder

class JupyterSolver:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "process_files_and_generate_response",
                "description": "Process files and generate responses based on the questions extracted from the files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dictionary": {
                            "type": "object",
                            "properties": {
                                "problem_file_types": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "List of file extensions."
                                },
                                "problem_files": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            #"document_id": {"type": "string"},
                                            "page_content": {"type": "string"},
                                            "metadata": {"type": "object"},
                                            "state": {"type": "object"}
                                            #"vector": {"type": "array", "items": {"type": "number"}}
                                        },
                                        "required": ["document_id", "content", "metadata", "vector"]
                                    },
                                    "description": "List of Document objects."
                                },
                                "tools": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"},
                                            "function": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "parameters": {"type": "object"}
                                                },
                                                "required": ["name", "description", "parameters"]
                                            }
                                        },
                                        "required": ["type", "function"]
                                    },
                                    "description": "Tools to be used in processing."
                                }
                            },
                            "required": ["problem_file_types", "problem_files", "tools"]
                        }
                    },
                    "required": ["dictionary"],
                },
            }
        }
    ],
    solver_tools = [
        {
            "type": "function",
            "function": {
                "name": "solve_problems",
                "description": ("Solve problems based on a list of extracted problem strings. This includes handling scenarios such as: "
                                "1. A single Jupyter notebook with a comprehensive list of questions. "
                                "2. A PDF with a full list of questions. "
                                "3. An optional PDF pointing to another PDF with the actual questions."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "problems": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of problem strings to be solved."
                        },
                        "problem_file_path": {
                            "type": "string",
                            "description": "Path to the file containing the list of problems."
                        },
                        "pointer_file_path": {
                            "type": "string",
                            "description": ("Path to the file pointing to the actual problems. "
                                            "This is applicable when the problems are spread across multiple documents, and a pointer file which lists the problems that need to be solved is uploaded.")
                        }
                    },
                    "required": ["problems", "problem_file_path"]
                }
            }
        }
    ]
    
    def convert_embeddings_to_documents(self, embeddings: List[List[float]], session_vector_store) -> List[Document]:
        documents = []
        if embeddings:
            for embedding in embeddings:
                summarized_embedding = session_vector_store.similarity_search_by_vector(embedding=embedding, k=1) # summarized_embedding is guaranteed to be a Document object
                documents.append(summarized_embedding)
        else:
            print("Current Embeddings None!")
            return
        return documents
    
    def is_response_code(self, completion) -> bool:
        """
        Heuristic to determine if a given text is likely to be code. This function could be
        implemented based on simple checks (e.g., presence of code-specific keywords or patterns)
        or more sophisticated classification models.
        """
        
        # First look for ``` ``` enclosing to determine whether or not the completion is of code type
        
        # Placeholder for actual implementation
        code_indicators = ['def ', 'class ', '# ', 'import ', 'from ', '=', '->']
        
        return completion, any(indicator in completion for indicator in code_indicators)
    
    async def send_for_completion(self, current_cell_content: str, persistent_context: List[Document], previous_context: List[Document]) -> Tuple[str, bool]:
        """
        Function to send cell content for autocompletion and determine if the response is code.
        """
        
        # printed example of the structure of a request with named parameters as it is made from App.tsx:
        # request base_url: http://localhost:8000/
        # request app: <fastapi.applications.FastAPI object at 0x2c97dca90>
        # request method: POST
        # request body: <bound method Request.body of <starlette.requests.Request object at 0x3162ae090>>
        # request query_params
        
        # How a request is constructed from App.tsx:
        # const [uploadedFileIds, setUploadedFileIds] = useState<string[]>([]);

        # const handleSendMessage = async (message: string) => {
        #     setInputValue("")
        #     closeCurrentEventSource(); // Close the previous event source if open

        #     const endpoint = isSolveMode ? '/solve' : '/rag/stream'; // Choose endpoint based on mode

        #     const messagePayload = {
        #     message: formatMessageForDisplay(message),
        #     isUser: true,
        #     ...(uploadedFileIds ? {file_embedding_keys: uploadedFileIds} : {}) // Include all file IDs that have been uploaded
        #     };
        
        #     setMessages(prevMessages => [...prevMessages, messagePayload]);
        
        #     const bodyPayload = {
        #     input: {
        #         question: message,
        #         file_embedding_keys: uploadedFileIds || {}, // Include the file_embedding_keys if they exists.
        #         session_id: sessionId || {}
        #     }
        #     };
        
        #     console.log("Sending message with payload:", bodyPayload);

        #     const newEventSource = await fetchEventSource(`${process.env.REACT_APP_BACKEND_URL}${endpoint}`, {
        #     method: 'POST',
        #     headers: {
        #         'Content-Type': 'application/json',
        #     },
        #     body: JSON.stringify(bodyPayload),
        #     onmessage(event) {
        #         if (event.event === "data") {
        #         handleReceiveMessage(event.data);
        #         }
        #     },
        #     onerror(error) {
        #         console.error('EventSource failed:', error);
        #         // Handle the error here, such as updating the UI to inform the user
        #     },
        #     });

        #     setCurrentEventSource(newEventSource as unknown as EventSource);

        #     if (uploadedFileIds) {
        #     setUploadedFileIds([]); // Reset the lastUploadedFileId after sending the message
        #     }
        # }
        
        # BASICALLY, we need to construct a new Request object the same way it is constructed when sending a request from the React side App.tsx with named parameters and then send it to an endpoint using the requests module.
        
        # How to use requests module to make direct request to a specific fastapi endpoint:
        # import requests

        # def test_function(request: Request, path_parameter: path_param):

        #     request_example = {"test" : "in"}
        #     host = request.client.host
        #     data_source_id = path_parameter.id

        #     get_test_url= f"http://{host}/test/{id}/"
        #     get_inp_url = f"http://{host}/test/{id}/inp"

        #     test_get_response = requests.get(get_test_url)
        #     inp_post_response = requests.post(get_inp_url , json=request_example)
        #     if inp_post_response .status_code == 200:
        #         print(json.loads(test_get_response.content.decode('utf-8')))
        
        # Use a modified version of the above to:
            # Make a direct request with current_cell_content as the problem parameter, persistent_context as the persistent_cell_context parameter and previous_context as the previous_cells_context parameter
                # Making a direct request to the /jupyter/stream endpoint will trigger the execution of solve_chain in rag_chain.py
                # I need to make a direct request to /jupyter/stream instead of just invoking the associated chain because there are some other necessary things done by
                # the api handler when sending a request to /jupyter/stream
                
        request_payload = {
            "current_cell_content": current_cell_content,
            "persistent_context": [doc.to_dict() for doc in persistent_context],
            "previous_context": [doc.to_dict() for doc in previous_context],
        }

        # Hypothetical URL construction; replace with actual server info
        url = "http://localhost:8000/jupyter/stream"
        
        # Asynchronously send the request using httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=request_payload)
            
        # Process the response
        # This is placeholder logic; replace with actual response processing
        if response.status_code == 200:
            response_data = response.json()
            completion = response_data.get("completion")
            # Further process completion to handle math symbols, LaTeX formatting, etc.
            # The `is_response_code` logic needs to be adjusted or integrated here
            is_code = self.is_response_code(completion)
        else:
            completion = "Error processing completion"
            is_code = False
        
        # Determining if the response is code
        completion, is_code = self.is_response_code(completion)
            # If completion type is markdown, do not modify completion and just return it again
            # If completion type is code, completion should be just the extracted formatted string within each ```enclosing```

        return completion, is_code

    async def execute_notebook_cell(self, nb: nbformat.NotebookNode, cell_index: int, kernel_name='tar_venv'):
        """
        Executes a specific cell in the notebook using nbclient and updates the cell with execution outputs.
        """
        client = NotebookClient(nb, kernel_name=kernel_name)
        await client.execute_cell(index=cell_index)
        
    async def embed_cell_content(self, cell_path: Path, cell_id):
        # Assume `FileEmbedder` can process individual cell files similarly to notebook files
        file_embedder = FileEmbedder(session_id=get_current_session_id(), pre_delete_collection=False)
        await file_embedder.process_and_embed_file(unique_id=cell_id, directory_path=str(cell_path.parent), embeddings=OpenAIEmbeddings(...), glob_pattern=str(cell_path.name))
    
    async def autocomplete_and_execute_cells(self, nb: nbformat.NotebookNode, current_notebook_documents: List[Document], session_vector_store) -> nbformat.NotebookNode:
        """
        Iterate through each cell of a Jupyter notebook, autocomplete each cell by sending it to a chat completions API,
        executes the cell if it's a code cell, and appends the completion or execution output as a new cell right below.
        """
        accumulated_cells = []
        accumulated_cell_ids = []
        
        # Directory for cells of a notebook
        notebook_dir = Path(f"./notebook_cells/{uuid.uuid4()}")
        notebook_dir.mkdir(parents=True, exist_ok=True)

        for cell_index, cell in enumerate(nb.cells):
            cell_id = str(uuid.uuid4())
            cell_path = notebook_dir / f"cell_{cell_index}_{cell_id}.txt"
            
            # Save initial cell content to a file
            with open(cell_path, 'w') as f:
                f.write(cell.source)
                
            # Embed current cell content
            await self.embed_cell_content(cell_path, cell_id)
            
            # Append processed cell to accumulated_cells
            accumulated_cells.append(cell)
            
            # Append processed cell_id to accumulated_cell_ids
            accumulated_cell_ids.append(cell_id)
            
            previous_context = []
            
            # Previous Context extraction as List[Document]Logic
            for cell_id in accumulated_cell_ids:
                embedding = get_embedding_from_manager(cell_id)
                documents = self.convert_embeddings_to_documents(embedding, session_vector_store)
                previous_context.extend(documents)
                
            # Determine cell type based on completion response, not input cell
            completion_response, is_code = await self.send_for_completion(current_cell_content=cell.source, persistent_context=current_notebook_documents, previous_context=previous_context)
                
        # Append the original cell and the completion response cell to the new cells list based on conditional logic
            append = []
            if is_code:
                # For code cells, create a new code cell for the completion response
                completion_cell = nbformat.v4.new_code_cell(completion_response) # This is correct!
                
                # Execute the completion cell, updating `nb` with new outputs
                completion_cell_index = cell_index + 1 
                
                completion_cell_id = str(uuid.uuid4())
                completion_cell_path = notebook_dir / f"cell_{completion_cell_index}_{completion_cell_id}.txt"
                
                # Save initial cell content to a file
                with open(completion_cell_path, 'w') as f:
                    f.write(completion_cell.source)
                    
                # Embed current cell content
                await self.embed_cell_content(completion_cell_path, completion_cell_id)
                
                # Append to accumulated_cell_ids
                accumulated_cell_ids.append(completion_cell_id)
                
                # Append to List that will be appended to accumulated_cells
                append.append(completion_cell)
                
                # This part likely requires a lot of debugging because the initial completion_cell made from an initial completion_response may cause errors.
                output = await self.execute_notebook_cell(nb, completion_cell_index)
                
                # Then generate a new markdown cell for the output.
                code_output_cell = nbformat.v4.new_markdown_cell(output)
                
                output_cell_index = cell_index + 2  # Index of the newly added output cell. Verify that this logic is correct.
                
                output_cell_id = str(uuid.uuid4())
                output_cell_path = notebook_dir / f"cell_{output_cell_index}_{output_cell_id}.txt"
                
                # Save initial cell content to a file
                with open(output_cell_path, 'w') as f:
                    f.write(code_output_cell.source)
                    
                # Embed current cell content
                await self.embed_cell_content(output_cell_path, output_cell_id)
                
                # Append to accumulated_cell_ids
                accumulated_cell_ids.append(output_cell_id)
                
                # Append cell to append
                append.append(code_output_cell)
            else:
                # For markdown or other types, create a new markdown cell
                completion_cell = nbformat.v4.new_markdown_cell(completion_response)
                
                completion_cell_index = cell_index + 1
                
                completion_cell_id = str(uuid.uuid4())
                completion_cell_path = notebook_dir / f"cell_{completion_cell_index}_{completion_cell_id}.txt"
                
                # Save initial cell content to a file
                with open(completion_cell_path, 'w') as f:
                    f.write(code_output_cell.source)
                    
                # Embed current cell content
                await self.embed_cell_content(completion_cell_path, completion_cell_id)
                
                # Append to accumulated_cell_ids
                accumulated_cell_ids.append(completion_cell_id)
                
                # Append completion_cell to append
                append.append(completion_cell)
            
            # Final addition of elements to accumulated_cells
            accumulated_cells.extend(append)

        # Update the notebook's cells with the new list including completions and execution outputs
        nb.cells = accumulated_cells
        
        return nb
    
    def process_files_and_generate_response(self, problems: Dict) -> List[str]:
        # To Do (Edge cases):
        
        # 1. Within a single jupyter notebook. In this case, the chat completions api must recognize that this jupyter notebook is the notebook with all the questions that need to be solved, make a list of all questions(while maintaining their name like 1. a.), and finally generate the appropriate cell(markdown or code) based on the individual response for each question string from the chat completions api. For code, the program should generate a code cell, write the code written by the chat completions api, and run it to generate output(until it succeeds running in the kernel without errors). When passing each question string to the chat completions api, we must also pass the previous questions and the code output for the previous questions so that the completions api can answer with context.

        # 2. Within a pdf that has the full list of questions that need to be solved. In this case, the chat completions api should initially recognize that this pdf is the document with the full set of questions, make a list of all questions(while maintaining their name like 1. a.), and finally generate the appropriate cell(markdown or code) based on the individual response for each question string from the chat completions api. For code, the program should generate a code cell, write the code written by the chat completions api, and run it to generate output(until it succeeds running in the kernel without errors). When passing each question string to the chat completions api, we must also pass the previous questions and the code output for the previous questions so that the completions api can answer with context.

        # 3. Within a pdf that identifies the questions that need to be solved and it refers to another pdf for the actual set of full questions. In this case, the chat completions api should recognize a) the full list of questions that need to be solved (e.g. Questions 1, 3, 9, 10 from name.pdf) go look for that pdf within the passed context, and generate a list of question strings - similar to the above. With the list of question strings, we can finally generate the appropriate cell(markdown or code) based on the individual response for each question string from the chat completions api. For code, the program should generate a code cell, write the code written by the chat completions api, and run it to generate output(until it succeeds running in the kernel without errors). When passing each question string to the chat completions api, we must also pass the previous questions and the code output for the previous questions so that the completions api can answer with context.

        # 4. If it’s not one of cases 1~3, raise a not implemented error.
        
        # So at the point of calling of this method, problems: Dict could be 1 of the 3 cases. The job of this method is to further process problems.get('problem_files', []) and extract the problems that need to be solved as a list of strings
        # Here is where we need to call the chat completions api once more to this time pass the problem_files as augment and make it return an identified function _ whose argument is a list of questions: List[str];
        
        problem_file_types = problems.get('problem_file_types', [])
        problem_files = problems.get('problem_files', [])
        
        for problem_document in problem_files:
            print(f"Test: {problem_document}")
        
        # Call
        
        # tools = problems.get('tools', [])  # Retrieve tools from the input dictionary
    
    def extract_questions(self, file_type, files):
        # Will be called by process_files_and_generate_response
        questions = []
        # if file_type == 'ipynb':
        #     # Process Jupyter Notebook to extract questions
        #     questions = extract_questions_from_notebook(files[0])
        # elif file_type == 'pdf':
        #     # Process PDF to extract questions
        #     questions = extract_questions_from_pdf(files[0])
        # elif file_type == 'pdf_list':
        #     # Process a list of PDFs to extract questions
        #     questions = extract_questions_from_multiple_pdfs(files)
        # else:
        #     # Handle unknown case or error
        #     raise ValueError("Unknown file type or case.")
        return questions
    
    def solve_problems(self, problems: Optional[List[str]], problem_file_path: Optional[str], pointer_file_path: Optional[str]):
        
        if not problems:
            print(f"No problems: {problems} to solve!")
            return
            
        for problem in problems:
            print(f"problem: {problem}")
        
        # Will be called by process_files_and_generate_response
        questions = []
        # if file_type == 'ipynb':
        #     # Process Jupyter Notebook to extract questions
        #     questions = extract_questions_from_notebook(files[0])
        # elif file_type == 'pdf':
        #     # Process PDF to extract questions
        #     questions = extract_questions_from_pdf(files[0])
        # elif file_type == 'pdf_list':
        #     # Process a list of PDFs to extract questions
        #     questions = extract_questions_from_multiple_pdfs(files)
        # else:
        #     # Handle unknown case or error
        #     raise ValueError("Unknown file type or case.")
        return questions
        
# Cookbook
# ```
# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
# def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             tools=tools,
#             tool_choice=tool_choice,
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e

# def ask_database(conn, query):
#     """Function to query SQLite database with a provided SQL query."""
#     try:
#         results = str(conn.execute(query).fetchall())
#     except Exception as e:
#         results = f"query failed with error: {e}"
#     return results

# def execute_function_call(message):
#     if message.tool_calls[0].function.name == "ask_database":
#         query = json.loads(message.tool_calls[0].function.arguments)["query"]
#         results = ask_database(conn, query)
#     else:
#         results = f"Error: function {message.tool_calls[0].function.name} does not exist"
#     return results

# messages = []
# messages.append({"role": "system", "content": "Answer user questions by generating SQL queries against the Chinook Music Database."})
# messages.append({"role": "user", "content": "Hi, who are the top 5 artists by number of tracks?"})
# chat_response = chat_completion_request(messages, tools)
# assistant_message = chat_response.choices[0].message
# assistant_message.content = str(assistant_message.tool_calls[0].function)
# messages.append({"role": assistant_message.role, "content": assistant_message.content})
# if assistant_message.tool_calls:
#     results = execute_function_call(assistant_message)
#     messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
# pretty_print_conversation(messages)
# ```

content=''
additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_LunBg1H7tJGy6vi7fqsdOJLl', 'function': {'arguments': '{"dictionary":{"problem_file_types":["pdf","js"],"problem_files":[{"document_id":"1","content":"Econ 573: Problem Set 4 Due date: 3/21\\n\\nSpring 2024\\n\\nThe problem set is due before the class, 9:30 AM. Instructions: give a justification to all of your answers.","vector":[],"tools":[]},{"document_id":"2","content":"Make sure to submit all the relevant files. For the empirical exercise you can find the data following this link:\\n\\nhttps://www.statlearning.com/resources-second-edition. Part I\\n\\nEx 3, 5, 6 from Chapter 5 of ISL. Part II\\n\\nEx 3, 9 from Chapter 7 of ISL. Part III\\n\\nEx 1, 5, 10 from Chapter 8 of ISL. 1","vector":[],"tools":[]},{"document_id":"3","content":"in the file.","vector":[-0.024693676262663923,-0.0055145548856154495,-0.023005337383361066,-0.026766351493786547,-0.006180281981974968,0.02726050009448575,-0.014906799046844802,-0.02833115415423721,0.0008300142673105025,0.01201054312499771,0.03461781646127582,0.006183713905790765,0.023156327130094422,-0.0038021946350778784,-0.019505121543257942,-0.008407379814969354,0.008846622291228838,0.021632703830142895,0.04150843402995741,-0.016801033797320734,-0.0011761752397078905,0.026752625661168573,-0.00824952715192701,-0.030801895295088012,-0.010027087668537127,-0.004811080352572592,0.029319451355635618,-0.020575775603009404,-0.006358724325266877,0.008434832411527907,-0.0079475476584603,-0.020589501435627378,-0.0170069287371712,-0.017926593050189305,-0.02212685056819688,-0.0075563469933476336,0.017803054968691893,-0.02555843538256492,0.017116739123405417,0.005517986343769942,0.02748012086695419,-0.009265276018561364,0.0027830143102890326,0.001086096203523312,-0.005504260045490665,0.03497469928521442,0.028715492368702196,0.003301183350900341,-0.020356154830540967,0.024927022867750333,-0.0010981067727253426,0.00962902362145417,-0.03294320155194572,-0.025681971601417112,0.008695632544495482,-0.010136898054771347,-0.013836144987093341,-0.009965318627788423,0.0033149098820102704,0.0023386241811365035,0.0009814328881055066,0.003767878889379685,-0.008935843928536098,0.010974204578113789,-0.0199031855903409,-0.005387586277286155,-0.004165942470801338,0.0026937931386430774,-0.005360133680727601,-0.01643042327811894,0.04979914240502878,0.026450646167701864,-0.004982659313894211,0.0008965012137756573,0.014687178274376364,-0.01496170423996191,-0.007803421293697235,-0.00877112741786216,0.022826894108746547,0.004426743069430217,-0.010823214831380433,-0.01947766801537678,-0.014412651377468208,-0.007700473823772003,0.02108365003632658,-0.0012551016876443876,0.017295182398019936,0.012786080607591446,-0.012195848384598607,0.009800603048437094,0.0355512066069119,0.05295619939116605,0.02389754816849801,-0.004773332915889253,0.017171644316522527,0.003846805220900856,0.015524484797718806,0.01984828039722379,0.00047184275481491547,-0.021358176001912126,0.0026165825361991533,0.024350517408698077,-0.03211962179254386,-0.008963296525094653,-0.035111965061975024,0.012120353511231929,0.02159152446964376,-0.01088498294080653,0.016389243917619804,-0.010466330144796613,-0.008874074887787393,0.011358541861256163,-0.012820396120458988,-0.041892772244422394,-0.014645998913877228,-0.008846622291228838,0.0069283673336714546,-0.01984828039722379,-0.024885845369896416,-0.009697655578511861,0.007082788538559303,0.02365047386814841,0.0398063711806818,-0.03357461592940552,0.015126420750635848,0.01224389066140673,-0.01145462641487241,-0.01907960583093904,0.007069062240280026,-0.019958090783458008,0.053203271828870434,0.015140146583253822,-0.001956002627071375,-0.004625774627310391,0.0042517321842928,0.00892898101222711,-0.027178143236132695,0.0011066857673575543,-0.028935113141170636,-0.01611471608938904,0.0238426429753809,0.016238254170886448,0.0015888232167780962,0.007350452053497169,-0.008723085141054037,0.004553711677759511,0.011722289464148971,0.015565663226895335,0.022360199035928507,-0.010027087668537127,0.0079475476584603,-0.030417557080623027,0.004515964241076172,-0.04397917703345343,-0.007666157845243157,0.011324226348388623,0.0004932901540839494,-0.0014421230255685093,-0.006454808878883124,0.00968392881457128,0.006087629352174519,0.029429261741869838,0.019505121543257942,-0.014865620617668275,0.021495439916027512,0.03675912458077874,0.02231901967542937,-0.001885655173767089,-0.014467556570585318,0.0051851227955901835,0.002208224114652716,0.03173528357807887,-0.034370740298281,0.01142030997068226,-0.009285865698810932,0.018008951771187578,0.005267481050927152,0.0032119621792543857,-0.0024535822202637666,-0.0014446967355997054,-0.02967633604221944,-0.013225323083850932,0.02550353018944781,0.020630680796126514,0.0017844234329191032,-0.00019860290209726202,0.027164415540869503,-0.014975431003902494,0.0032754464834190327,-0.0026268773763239374,0.03489234056421615,0.01654023366435316,0.00571015498534113,-0.03176273896860526,-0.683461632061188,-0.008386790134719784,-0.018626636590738974,-0.03003322072880326,0.023444578928297943,0.02142680889029243,0.01788541368969017,0.00892898101222711,-0.026642815274934357,0.007117104517088149,-0.0289625666690518,0.004876280385814486,0.012923343590384221,-0.0032994676218230943,-0.003110730671237052,-0.010850667427938988,-0.005469944532623124,-0.011591889397665183,-0.025476076661566645,0.011468352247490383,-0.0003287886043098038,0.014906799046844802,-0.007103378218808872,0.0039257315524220265,0.020630680796126514,0.026340835781467644,0.0037781734966738168,-0.0211934604225608,0.02437797093657924,0.010706540597514618,-0.015085241390136712,-0.005425333713969494,0.00835933753816123,-0.012120353511231929,-0.007583799589906189,0.006482261475441679,-0.036237524314843596,0.0029768989137681193,0.03733562817718578,0.023238683988447476,-0.0004486795682609718,-0.0044095848473351415,0.010287887801504702,-0.014048903774575402,0.003265151876124901,-0.0062797979937457065,0.012840985800708556,0.005562597162423572,0.03003322072880326,-0.008620137671128804,0.007981863171327841,-0.015702926209688106,0.0012653964113538455,-0.009780013368187526,-0.018008951771187578,0.001827318289664834,0.04156334108571974,-0.020109080530191364,0.014385198780909653,0.005672407548657791,-0.00391200525414275,0.010534962101854305,-0.007981863171327841,-0.022003314349344688,-0.002506771917134282,-0.007851463104844053,-0.02844096454047143,0.0017638339855001871,0.016650044050587377,-0.00973883400768839,0.012264480341656298,0.007508305182200815,-0.0030283726487307355,-0.0071994623067638125,0.005892029252448838,0.0021756240980317695,0.005500828587336172,-0.01233311136739138,-0.0021790555561862625,0.005150806817061338,0.007288683478409768,-0.01290275484145726,-0.0360453552076111,0.00010948896351863477,0.019820826869342625,-0.012291932938214853,-0.008675043795568523,-0.0059778185002789955,-0.023211232323211532,-0.023883822335880038,0.021975860821463522,0.009876097921803773,-0.007371041733746737,-0.02614866667423515,-0.004735585479205914,0.007707336740080989,-0.0016119863742282082,-0.006410198060229494,0.015442126076720533,-0.013341997317716747,-0.007700473823772003,-0.006880325056863331,-0.003103867289266761,-0.00995159279517045,0.029209639106756183,0.0277821003604209,-0.024872117674633223,-0.005291501723669909,0.030060672394039208,-0.024226981189845884,-0.01088498294080653,-0.005452786310528049,-0.010953614897864221,-0.006698451255416927,-0.013650839727492443,-0.0328333911657115,0.013067470420808589,0.02218175576131399,0.017459897977371266,-0.005631228653819959,0.020479691049393157,-0.0018376130133742922,-0.0035276679710003742,-0.0069626833122003,0.021811145242112195,0.023664201563411598,0.05048546011296048,-0.02061695496350854,-0.022168029928696015,0.0028464986144536796,0.0026182982652763996,0.002934004057022388,0.029484166934986948,-0.014316566823851962,0.003088425261910237,0.00333378360035194,0.03928477091474665,0.0039257315524220265,0.032366697955538676,-0.007494578418260233,-0.026395740974584754,-0.0030558252452892905,-0.017569708363605482,0.006976409610479578,-0.00310558325117466,-0.027891912609300337,-0.01770697227772087,0.007782831613447667,-0.02597022526226585,0.006159692767386703,0.007000430748883639,-0.009320181211678474,-0.006832282780055208,0.012875301313576098,-0.00920350697781266,-0.014783261896670001,-0.0015819600676384574,-0.02378773778226379,-0.0007300694396857778,-0.0068837569806791295,-0.0011650226514598092,0.03137840075414027,-0.029703787707455385,0.014275387463352825,-0.022648452696777246,-0', 'name': 'process_files_and_generate_response'}, 'type': 'function'}]}
