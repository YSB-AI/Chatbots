import chainlit as cl
import os
from llama_index.core.postprocessor import LLMRerank
from PydanticAI.chainlit_app.prompts import * 

host = os.getenv("host", None)
user = os.getenv("user", None)
password = os.getenv("password", None)
port = os.getenv("port", None)
database= os.getenv("database", None)
table_name= os.getenv("table_name", None)
decision_agent_model = os.getenv("decision_agent_model", None)
tool_agent_model = os.getenv("tool_agent_model", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from llama_index.core import QueryBundle
import time

# Add missing import
import json
from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional
from doc_sources_management import *

DOCUMENT_ROOT_PATH = "./DOCUMENTS/"
os.makedirs(DOCUMENT_ROOT_PATH, exist_ok=True)
print(f"Directory '{DOCUMENT_ROOT_PATH}' created or already exists.")

class AIAgent():
    def __init__(
            self,
            retriever=None,
            llm=None,
            agent_retry = 10,
            **kwargs
        ):
        self.llama_model = OpenAIModel(
                model_name='llama3.2', 
                provider=OpenAIProvider(base_url='http://localhost:11434/v1')
            )
        self.agent_retry = agent_retry
        self.task_list = None
        self.task_def = None
        self.sources_doc = []
        self.task_running = ""
        self.interaction_agent_history = None

        self.doc_creation_revision_counter = 1
        self.doc_writer_revision_counter = 1
        self.final_output_revision_counter = 1

        self.model = 'google-gla:gemini-2.0-flash'
        
        self.retriever = retriever
        self.doc_planning_json = None
        self.search_result = None
        self.sections_max_review = 1

        # Define the tool before using it in agent initialization
        self.doc_creation_supervisor_tool = Tool(
            self.doc_creation_supervisor,
            name="doc_creation_supervisor_tool",
            description="If the user wants to create a new document or generate new content this tool will supervise the document creation process. All the validations and decisions will be done here.",
        )

        self.retrieve_data_tool = Tool(
            self.retrieve_data,
            name="retrieve_data_tool",
            description="Search and retrieve information from the knowledge base",
        )

        self.doc_planning_tool = Tool(
            self.doc_planning,
            name="doc_planning_tool",
            description="Performs a plan for the document creation and return its structure, sections, subsections and their objectives.",
        )

        self.doc_creation_tool= Tool(
            self.doc_creation_process,
            name="doc_creation_tool",
            description="Receives the document plan and the search result and creates a complete document accordingly.",
        )

        self.doc_update_tool = Tool(
            self.doc_update_process,
            name="doc_update_tool",
            description="Receives the document title/subject and section to be changed, the changes to be applied to the section's text, and perform the changes accordingly.",    
        )

        self.interactor_agent = Agent(
            self.model,
            retries = agent_retry,
            system_prompt = INTERACTOR_AGENT_SYSTEM_PROMPT.replace("$LANGUAGE$", str(cl.user_session.get('language'))) ,
            tools=[self.retrieve_data_tool, self.doc_creation_supervisor_tool]
                    
        )

    
    async def retrieve_data(self, query: str) -> str:
        """Search and retrieves chunks of information from the knowledge base."""
        print(f"Tool 'retrieve_data' called with query: {query}")
        await self.manage_tasks(self.task_list, self.task_def,task_running="retrieve_data", status = cl.TaskStatus.RUNNING , revision_counter = None)

        if not self.retriever:
            return "No retriever configured."
            
        # Get search results directly using the query
        query_result_list = await self.retriever.aretrieve(QueryBundle(query))
        
        sources = []
        ids_retrieved = []
        sources_elements = []
        print("Search result : ",len(query_result_list))
        if len(query_result_list) > 0:
            for result in query_result_list:
                chunk_id = result.node.id_
                metadata = result.node.metadata
                
                if chunk_id not in ids_retrieved:#result.score > 0.3 and 
                    ids_retrieved.append(chunk_id)
                    sources.append(result.node.text)
                    # Sources to be presented
                    doc_name = metadata["file_path"].split("/")[-1].replace(".pdf", "")
                    sources_elements.append(cl.Pdf(name=doc_name, path = metadata["file_path"], display="inline") )

        self.sources_doc = sources_elements
        str_sources = '\n'.join(sources)
        print(f"Retrieved sources: {len(sources)}")
        await self.manage_tasks(self.task_list, self.task_def,task_running="retrieve_data", status = cl.TaskStatus.DONE , revision_counter = None)

        return str_sources

    async def doc_creation_supervisor(self, query: str) -> str:
        """If the user is asking about existing documents, specific information within documents, or mentions authors, works, papers, this tool answer to the user's questions using information from the knowledge base."""

        history  = cl.user_session.get("most_recent_history")
        print(f"Tool 'doc_creation_supervisor' called with the search result: {query}")

        input_prompt = f"""Here you have the conversation history to be used as a context : {history}
        Here you have the user input: {query}
        """
        doc_creation_agent = Agent(
            self.model,
            deps_type=str,
            retries = self.agent_retry,
            system_prompt=DOC_CREATION_SYSTEM_PROMPT ,
            tools=[self.retrieve_data_tool, self.doc_planning_tool, self.doc_creation_tool, self.doc_update_tool],
        )

        result = await doc_creation_agent.run(input_prompt)
        print("Document creation output : ",result)

        return result
    

    async def doc_planning(self, document_subject : str) -> str:
        """Performs a plan for the document creation based on the doc subject inserted by the user, and return its structure, sections, subsections and their objectives."""

        await self.manage_tasks(self.task_list, self.task_def,task_running="doc_planning", status = cl.TaskStatus.RUNNING , revision_counter = None)

        print(f"Tool 'doc_planning' called with the document subject: {document_subject}")

        self.search_result = await self.retrieve_data(document_subject)
        inputs = f"""Here you have the document subject : {document_subject}
        Here you have the contexto about the subject : {self.search_result}
        Now plan out the structure of the dissertation in detail, defining the sections/subsections names and objective according to the document's title and objectives."""

        doc_planning_agent = Agent(
            self.model,
            retries = self.agent_retry,
            system_prompt=DOC_PLANNING_AGENT_SYSTEM_PROMPT.replace("$LANGUAGE$", str(cl.user_session.get('language'))) ,
        )
        result = await doc_planning_agent.run(inputs)
        output_json_str = result.data.split("[JSON]")[-1].replace("json","").replace("```", "").replace("```","").strip()
        self.doc_planning_json = json.loads(output_json_str)
        print("doc_planning_json : ", self.doc_planning_json)
        
        cl.user_session.set("doc_planning_json", self.doc_planning_json)
        
        await self.manage_tasks(self.task_list, self.task_def,task_running="doc_planning", status = cl.TaskStatus.DONE , revision_counter = None)
        return f"Present the following document plan to the user in human readable and organized way : {self.doc_planning_json}"
    
    def get_available_sections(self, doc_planning_json, sections_review_control, sections_max_review):
       
        available_sections = []
        available_sections_to_process = []
        for details in doc_planning_json["sections"]:
            if details["title"] in sections_review_control:
                try:
                    print(f" {details['title']} -> {sections_review_control[details['title']]}/{sections_max_review}")
                    if sections_review_control[details["title"]] < sections_max_review:
                        available_sections.append(details["title"])
                        available_sections_to_process.append(details)
 
                except Exception as e:
                    print(f"Failed to append/evaluate {details['title']} final doc : {e}")

            else:
                available_sections.append(details["title"])
                available_sections_to_process.append(details)
        
        if len(available_sections) == 0:
            available_sections = ["FINISH"]
            
        return available_sections, available_sections_to_process
 

    async def doc_update_process(self, doc_subject : str, doc_sections_title_list : list,  doc_changes : str) -> str:
        """Receives the document title/subject and sections to be changed, the changes to be applied to the sections text, and perform the changes accordingly."""
        print(f"Tool 'doc_update_process' called with the :\nDoc changes: {doc_changes}\nDoc subject : {doc_subject}\nDoc section : {doc_sections_title_list}")
        await self.manage_tasks(self.task_list, self.task_def,task_running="doc_update_process", status = cl.TaskStatus.RUNNING , revision_counter = None)

        doc_subject = ' '.join(doc_subject.replace("\n"," ").split()).strip().replace("  "," ")
        # Getting doc construction dir 
        doc_construction_dir = f"{DOCUMENT_ROOT_PATH}/{(' '.join(doc_subject.split())).strip().replace(' ','_').replace(':','_').replace('.','_').replace('\n','')}_sections"
        print("Document construction directory : ",doc_construction_dir)

        if not os.path.isdir(doc_construction_dir):
            return f"Directory {doc_construction_dir} does not exist. The file name must be wrong or wrongly formatted."

        # Read planning for the specified document
        planning_file_path = doc_construction_dir+"/doc_planning.json"
        with open(planning_file_path, 'r') as file:
            doc_planning = json.load(file)

        # If previously updated, then read it. If not, then read the original document
        complete_doc_txt_path = f"{doc_construction_dir}/{doc_subject.strip().replace(' ','_').replace(':','_').replace('.','_')}.txt"
        if len(doc_sections_title_list)> 0 :
            complete_doc = ""
            for sec in doc_sections_title_list:
                sec_doc_txt_path = f"{doc_construction_dir}/{sec.strip().replace(' ','_').replace(':','_').replace('.','_')}_refined.txt"
                if os.path.exists(sec_doc_txt_path) == False:
                    sec_doc_txt_path = f"{doc_construction_dir}/{sec.strip().replace(' ','_').replace(':','_').replace('.','_')}.txt"

                print("Document path : ",sec_doc_txt_path)
                with open(sec_doc_txt_path, "r") as f:
                    complete_doc = complete_doc + "\n\n\n"+sec+"\n\n\n"+f.read()
        else:
            
            print("Document path : ",complete_doc_txt_path)
            with open(complete_doc_txt_path, "r") as f:
                complete_doc = f.read()

        print("Document read sucessfully. Let's update it.")
        output = await self.rewrite_doc(doc_planning, complete_doc_txt_path, doc_sections_title_list, doc_construction_dir, doc_subject, doc_changes, complete_doc)

        await self.manage_tasks(self.task_list, self.task_def,task_running="doc_update_process", status = cl.TaskStatus.DONE , revision_counter = None)
        return output

    async def rewrite_doc(self, doc_planning, complete_doc_txt_path, doc_sections_title_list, doc_construction_dir, doc_subject, doc_changes, complete_doc):
        
        """
        This function takes the document planning, the document subject, the sections to be updated, the construction directory, the document changes and the complete document text.
        It applies the document changes to the specified sections and updates the pdf and txt files.
        The function returns a message indicating if the document changes were applied successfully or not.
        """
        try:

            print("Initializing new pdf...")
            pdf = init_pdf(doc_title = doc_planning["doc_title"])

            print("Initializing agent ...")
            #print("Section content :",complete_doc)
            doc_update_agent = Agent(
                    self.model,
                    deps_type=str,
                    retries = self.agent_retry,
                    system_prompt= DOC_UPDATE_SYSTEM_PROMPT,
                    tools=[self.retrieve_data_tool], 
                )
            complete_text = ""
            for section_title in doc_planning['ordered_sections']:
                if len(doc_sections_title_list)> 0 and section_title not in doc_sections_title_list:
                    sec_doc_txt_path = f"{doc_construction_dir}/{section_title.strip().replace(' ','_').replace(':','_').replace('.','_')}_refined.txt"
                    if os.path.exists(sec_doc_txt_path) == False:
                        sec_doc_txt_path = f"{doc_construction_dir}/{section_title.strip().replace(' ','_').replace(':','_').replace('.','_')}.txt"

                    print("Document path : ",sec_doc_txt_path)
                    with open(sec_doc_txt_path, "r") as f:
                        section_doc = f.read()
                        complete_text = complete_text + "\n\n\n"+section_title+"\n\n\n"+section_doc+"\n---------------------------------------"

                elif section_title in doc_sections_title_list  or len(doc_sections_title_list)== 0:

                    complete_prompt = f"""Here you have the document subject : {doc_subject}
                    Here you have the documents sections already written : {complete_doc}
                    Here you have the document changes to apply: {doc_changes}
                    Now check if the changes are related to the section : {section_title}. 
                    If yes, then apply them to the specified section and output only the updated section content without your any extra comments or explanation.
                    If no, then extract the exact content of the specified section and output the response without your any extra comments or explanation.
                    Your final output should not include the main section title.
                    """
                    writer_output = await doc_update_agent.run(complete_prompt)
                    section_doc = writer_output.data.replace("```text","").replace("```","")
                    print("Changes applied : ",section_doc)

                    #writing the new changes to a refined version
                    sec_doc_txt_path = doc_construction_dir+"/"+section_title.replace(' ','_').replace(':','_').replace('.','_')+"_refined.txt"
                    with open(sec_doc_txt_path, 'w') as file_object:
                        file_object.write(section_doc)

                    complete_text = complete_text+"\n\n\n"+section_title+"\n\n\n"+section_doc+"\n---------------------------------------"
                
                #Update PDF
                pdf = write_pdf(pdf, section_title, is_title = True)
                pdf = write_pdf(pdf, section_doc, is_title = False)

            # Write the final pdf
            refined_doc_path = f"{DOCUMENT_ROOT_PATH}/REFINED_{doc_planning['doc_title'].strip().replace(' ','_').replace(':','_').replace('.','_')}.pdf"
            pdf.output(refined_doc_path, "F")

            sources_elements = [cl.Pdf(name=doc_planning['doc_title'], path = refined_doc_path, display="inline") ]
            self.sources_doc = sources_elements

            with open(complete_doc_txt_path, 'w') as file_object:
                file_object.write(complete_text)

            return f"Changes performed to the document {doc_subject}. Please check the new document in the sources section."
            
        except Exception as e:
            print(f"Failed to write the structured document : {e}")
            return f"Something went wrong. Please try again."


    async def doc_creation_process(self, doc_subject : str,  doc_planning : str) -> str:
        """Receives the document plan and the document subject and creates a complete document accordingly.."""
        print(f"Tool 'doc_creation_process' called with the :Doc plan: {doc_planning}\nDoc subject : {doc_subject}")
        
        
        if self.search_result is None: 
            self.search_result = await self.retrieve_data(doc_subject)

        sections_review_control = {}
        doc_supervisor_outputs = {}
        last_section_id = None
        docs_written_sections = []
        final_doc= ""

        retry = 5
        finished = False
        output_format_review = False
        review_message  = ""
        print_frequency = 0

        doc_writer_agent = Agent(
            self.model,
            deps_type=str,
            retries = self.agent_retry,
            system_prompt= DOC_WRITER_SYSTEM_PROMPT,
            tools=[self.retrieve_data_tool], 
        )

        while True: #for round in range(3):
            await self.manage_tasks(self.task_list, self.task_def,task_running="doc_creation_process", status = cl.TaskStatus.RUNNING , revision_counter = None)#self.doc_creation_revision_counter
            print("\n-------------------SUPERVISOR-------------------\n")
            for i in range(retry):
                try:

                    available_sections, available_planning_to_follow = self.get_available_sections(self.doc_planning_json, sections_review_control, self.sections_max_review)

                    if 'next_section' in doc_supervisor_outputs:
                        last_section_id = doc_supervisor_outputs['next_section']
                        print(f"The supervisor is evaluating the {last_section_id} and deciding what to do next...")

                        last_section_result = doc_supervisor_outputs[last_section_id]['results']
                        other_sections = []
                        for section in self.doc_planning_json['ordered_sections']:
                            if section in doc_supervisor_outputs and section != last_section_id:
                                other_sections.append(f"Section '{section}' : {doc_supervisor_outputs[section]['results']}")

                        prompt =  f"""- Here you have the document subject/title : {doc_subject}
                        - Here you have the document planning : {available_planning_to_follow}

                        - Here you have the others sections already written : {'\n'.join(other_sections)}
                        Consider this when evaluating the last section written, to avoid repetition among sections and to guide the writer to create a coherent document.

                        - Here you have the results for the last section written '{last_section_id}', that you MUST evaluate and provide an observation if refinement is needed : {last_section_result}
                        PROVIDE OBSERVATION ONLY FOR THE LAST SECTION WRITTEN.
                        
                        Now, perform your task and output the next section to be written, its respective objectives and subsections, and provide the observation for refinement of the last written section (if any).
                        THE ONLY AVAILABLE SECTION TO BE SELECT NOW ARE (the rest of the sections already finished being created) : {available_sections}
                        Do not forget that your output MUST follow the specified JSON format .
                        """
                    else : 
                        last_section_id = None
                        prompt = f"""- Here you have the document subject/title : {doc_subject} 
                        - Here you have the document planning : {doc_planning}
                        
                        Now, perform your task and output the next section to be written, its respective objectives and subsections, and provide the observation for refinement of the last written section (if any).
                        Do not forget that your output MUST follow the specified JSON format .
                        """
                    

                    if output_format_review:
                        prompt = prompt +"\n" +review_message
                        output_format_review = False
                        review_message = ""
                    
                    if print_frequency < 4:
                        print("\n--------------------------------")
                        print("Supervisor prompt : ",prompt)
                        print("--------------------------------\n")
                        print_frequency = print_frequency + 1

                    
                    print("Available sections : ",available_sections)
                    print("")

                    doc_writer_supervisor_agent = Agent(
                        self.model,
                        retries = self.agent_retry,
                        system_prompt=DOC_WRITER_SUPERVISOR_SYSTEM_PROMPT.replace("$AVAILABLE_SECTIONS$", str(available_sections)) ,
                    )
                    
                    supervisor_result_str = await doc_writer_supervisor_agent.run(prompt)
                    supervisor_result_str = supervisor_result_str.data.split("[JSON]")[-1].replace("json","").replace("```", "").replace("```","").strip()
                    supervisor_result = json.loads(supervisor_result_str)
                    print("\n------------------")
                    print("Supervisor result : ",supervisor_result)
                    print("")

                 
                    if ("FINISH" in available_sections)  or (len(available_sections) == 0):
                        doc_supervisor_outputs, last_section_id, final_doc = self.manage_sections_review(doc_supervisor_outputs, last_section_id, supervisor_result)
                        finished = True
                        await self.manage_tasks(self.task_list, self.task_def, task_running="doc_creation_process", status = cl.TaskStatus.DONE , revision_counter = None)#self.doc_creation_revision_counter
                        break
               
                    doc_supervisor_outputs, last_section_id, final_doc = self.manage_sections_review(doc_supervisor_outputs, last_section_id, supervisor_result)
                    await self.manage_tasks(self.task_list, self.task_def, task_running="doc_creation_process", status = cl.TaskStatus.DONE , revision_counter = None)#self.doc_creation_revision_counter
                    docs_written_sections, sections_review_control, doc_supervisor_outputs, doc_writer_agent = await self.doc_writer_process(doc_subject, supervisor_result,  doc_supervisor_outputs, docs_written_sections, sections_review_control, doc_writer_agent)
            
                    break
                except Exception as e:
                    print("Documet failed to be created : ",e)
                    print("Retry : ",i)
                    if i < retry-1:
                        review_message = f"\n An error occurred in your last output : {e}\n. Make sure to fix it in your next output."
                        print(f"Retry counter {i} - Retrying in 30 seconds...")
                        time.sleep(30)
                    else:
                        await self.manage_tasks(self.task_list, self.task_def,task_running=self.task_running, status = cl.TaskStatus.FAILED , revision_counter = None)
                        break

            if i == retry-1 or finished:
                break
                

        await self.write_pdf_file(final_doc, doc_supervisor_outputs)

        # self.doc_planning_json = None
        # self.search_result = None

        time.sleep(1)
        return final_doc

    async def write_pdf_file(self, final_doc, doc_supervisor_outputs):
        
        #https://stackoverflow.com/questions/56761449/unicodeencodeerror-latin-1-codec-cant-encode-character-u2013-writing-to
        doc_path = f"{DOCUMENT_ROOT_PATH}/{self.doc_planning_json['doc_title'].strip().replace(' ','_').replace(':','_').replace('.','_')}.pdf"
        doc_construction_dir = f"{DOCUMENT_ROOT_PATH}/{self.doc_planning_json['doc_title'].strip().replace(' ','_').replace(':','_').replace('.','_')}_sections"
        

        print(f"Creating file : {doc_path}")
        print(f"Construction directory : {doc_construction_dir}")
        os.makedirs(doc_construction_dir, exist_ok=True)

        print("final_doc : ",final_doc)
        

        extractor_agent = Agent(
            self.model,
            retries = 10,
            # Clearer system prompt
            system_prompt = (
                "You are an expert in formmating PhD thesis sections while keeping all the information exactly the same. \n"
                "You will be provided with a PhD thesis section. and your task is to format it in a way that is suitable for a professional for a PhD thesis. \n"
                "Output only the response without your any extra comments or explanation. "
            ),                            
        )

        complete_text = ""
        try:
            pdf = init_pdf(doc_title = self.doc_planning_json["doc_title"])

            print("Writing content section by section...")
            for section_title in self.doc_planning_json['ordered_sections']:
                if section_title in doc_supervisor_outputs:
                    print("\n--------------------------------------")
                    print(f"Writing section : {section_title}")
                    section_doc_path = doc_construction_dir+"/"+section_title.replace(' ','_').replace(':','_').replace('.','_')+".txt"
                    pdf = write_pdf(pdf, section_title, is_title = True)

                    input_str = f"""Here you have the section '{section_title}' of the PhD : {doc_supervisor_outputs[section_title]['results']}.
                            Format the section content as follows : 
                            1 - REMOVE THE SECTION MAIN TITLE '{section_title}' FROM THE TEXT CONTENT;
                            2 - Apply capital letters to all the subsections titles and remove any ** or # around them;
                            3 - DO NOT Apply bold font do any text.
                            4 - If tables or equations exists, format them properly for a PhD ;
                            5 - When bullet points exists, format them properly using '-';
                            6 - The output should be in {cl.user_session.get('language')}.
                            Output only the response without your any extra comments or explanation. """
                    
                    result = await extractor_agent.run(input_str)
                    text_data = result.data.replace("**","").replace('```','').replace("\\","").replace("#","")#.encode('latin-1', 'replace').decode('latin-1')
                    complete_text = complete_text+"\n\n\n"+section_title+"\n\n\n"+text_data+"\n---------------------------------------"

                    #Saving the section content to a file
                    with open(section_doc_path, 'w') as file_object:
                        file_object.write(text_data)
                    
                    pdf = write_pdf(pdf, text_data, is_title = False)

            pdf.output(doc_path, "F")
            sources_elements = [cl.Pdf(name=self.doc_planning_json['doc_title'], path = doc_path, display="inline") ]

            self.sources_doc = sources_elements
            
            # Save the planning file
            planning_file_path = doc_construction_dir+"/doc_planning.json"
            with open(planning_file_path, 'w') as json_file:
                json.dump(self.doc_planning_json, json_file, indent=4)

            # Save the completedocument text file
            doc_txt_file = doc_construction_dir+"/"+self.doc_planning_json['doc_title'].strip().replace(' ','_').replace(':','_').replace('.','_')+".txt"
            with open(doc_txt_file, 'w') as file_object:
                file_object.write(complete_text)
            
        except Exception as e:
            print(f"Failed to write the structured document : {e}")


    def manage_sections_review(self, doc_supervisor_outputs, last_section_id, supervisor_result):
            
        """
        Manage the review process of document sections based on the supervisor's evaluation.

        This function updates the document supervisor outputs with observations for the last section 
        and sets the next section to be written or refined. It appends completed sections to the final 
        document if the document creation process is finished.

        Args:
            doc_supervisor_outputs (dict): Contains outputs from the document supervisor, including section 
                                        observations and objectives.
            last_section_id (str): The ID of the last section that was reviewed.
            router (Router): An instance containing the next section details, objectives, subsections, and observations.
            doc_details (dict): Contains the document's details, including ordered sections.
            final_doc (str): The current state of the final document being constructed.

        Returns:
            tuple: Updated doc_supervisor_outputs, last_section_id, doc_details, and final_doc.
        """

        if "next_section" in doc_supervisor_outputs:
            if last_section_id in doc_supervisor_outputs:
                doc_supervisor_outputs[last_section_id]['observation'] = supervisor_result['observation']
                print("\n--------------------------------------")
                print(f"Supervisor last section new observation '{last_section_id}' : {doc_supervisor_outputs[last_section_id]['observation']}")
                print("--------------------------------------")

        final_doc = ""
        doc_supervisor_outputs["next_section"] = supervisor_result["next_section"]
        doc_supervisor_outputs["objectives"] = supervisor_result["objectives"]
        doc_supervisor_outputs["subsections"] = supervisor_result['subsections']

        if supervisor_result["next_section"] == "FINISH":
            for section_title in self.doc_planning_json['ordered_sections']:
                if section_title in doc_supervisor_outputs:
                    final_doc = final_doc + doc_supervisor_outputs[section_title]['results']+"\n\n"
        
        return doc_supervisor_outputs, last_section_id,  final_doc
    

    async def doc_writer_process(self, doc_subject, supervisor_result, doc_supervisor_outputs, docs_written_sections, sections_review_control, doc_writer_agent):
        """
        This function is responsible for the process of writing a section of the document. The function manages the history of the section, including the user's inputs and the supervisor's observations.
        The function uses the supervisor's output to generate a prompt for the writer, which includes the section to be written, the objectives, subsections, and the supervisor's observations.
        The function also appends the writer's output to the section history and updates the section's observation.
        If the section has been written before, the function will include the previous attempts in the prompt.
        The function will also include the critiques for refinement if the section has been written before.
        If the section has been written before the function will not include the context of the other sections of the document.
        The function will use the doc_writer_agent to generate the section based on the prompt.
        The function will return the updated section history, the updated documents written sections, and the updated sections review control.
        """
        section_id = supervisor_result["next_section"]
        doc_writer_history = section_id+"_history"
        print(f"\n----------------DOC WRITER PROCESS FOR SECTION '{section_id}'----------------------\n")


        await self.manage_tasks(self.task_list, self.task_def,task_running="doc_writer_process", status = cl.TaskStatus.RUNNING , revision_counter = None)#self.doc_writer_revision_counter

        if doc_writer_history not in doc_supervisor_outputs:
            prompt= f"""-Here you have the complete document planning : {self.doc_planning_json}
            - Here you have the document subject/title : {doc_subject}
            - Section to be written : {supervisor_result['next_section']}
            - Objectives of the section : {supervisor_result['objectives']}
            - Subsections and their objectives : {supervisor_result['subsections']}
            - Here you have the knowledge base search result: {self.search_result}"""
            doc_supervisor_outputs[doc_writer_history] = []#["user : "+ prompt]

        other_sections = []
        for section in self.doc_planning_json['ordered_sections']:
            if section in doc_supervisor_outputs and section != section_id:
                other_sections.append(f"Section '{section}' : {doc_supervisor_outputs[section]['results']}")

        if section_id in  docs_written_sections:
            prompt =  f"""(Section '{section_id}') - Here you have the observations about your last output to be used for refinement: {doc_supervisor_outputs[section_id]['observation']}
            """

        complete_prompt = f"""{prompt}
        - Here you have the other sections from the document that were already written: {'\n'.join(other_sections)}
        - Here you have the section creation history: {'\n'.join(doc_supervisor_outputs[doc_writer_history])}

        Now perform your tasks and generate the best section as possible, providing the highest quality and most detailed information as possible in the section.
        Format the every section or subsection title to be capital letter to enphasize that it is a beginning of a section/subsection.
        Output only the response without your any extra comments or explanation. 
        """
        writer_output = await doc_writer_agent.run(complete_prompt)
        writer_output = writer_output.data
        
        print(f"Writer output for '{section_id}' :  ",writer_output)
        print("------------------------")
        doc_supervisor_outputs[doc_writer_history].append("user : "+ prompt)
        doc_supervisor_outputs[doc_writer_history].append(writer_output)

        # Save results and reseting observation as it was already considered in the prompt above.
        doc_supervisor_outputs[section_id] = {"results" : writer_output, "observation": ""}

        if section_id not in docs_written_sections:
            docs_written_sections.append(section_id)

        if section_id in sections_review_control:
            sections_review_control[section_id] = sections_review_control[section_id] +1
            if sections_review_control[section_id] > self.sections_max_review:
                sections_review_control[section_id] = self.sections_max_review
        else:
            sections_review_control[section_id] = 1

        await self.manage_tasks(self.task_list, self.task_def,task_running="doc_writer_process", status = cl.TaskStatus.DONE , revision_counter = None)
        self.doc_writer_revision_counter = self.doc_writer_revision_counter + 1

        time.sleep(1)
        return docs_written_sections, sections_review_control, doc_supervisor_outputs, doc_writer_agent

    async def call_interactor_agent(self, input_prompt, result, refinement_feedback):
        """
        Asynchronously calls the interactor agent with the provided input prompt.

        This function is responsible for managing the interaction with the interactor agent,
        including handling any refinement feedback and ensuring the task status is updated
        appropriately. If the interactor agent fails to produce a result, the function will 
        attempt to notify the user about the server-side issue.

        Parameters
        ----------
        input_prompt : str
            The prompt to be sent to the interactor agent for processing.
        result : Any
            The current result from the interactor agent, if any.
        refinement_feedback : str or None
            Feedback for improving the previous response, if applicable.

        Returns
        -------
        Any
            The result returned by the interactor agent after processing the input prompt.
        """

        await self.manage_tasks(self.task_list, self.task_def,task_running="Thinking", status = cl.TaskStatus.RUNNING , revision_counter = self.final_output_revision_counter)
        
        try:
            if refinement_feedback is not None:
                prompt = prompt + f"""\n\n
                FEEDBACK FOR IMPROVEMENT:\n
                Your previous response didn't meet quality standards. Please address these issues: {refinement_feedback}
                "Make sure to completely rewrite your response addressing all feedback points."""

            if self.interaction_agent_history is not None:
                print("\n------------------------\nAgent conversation history : ", self.interaction_agent_history.all_messages(),"\n----------------------")

                result = await self.interactor_agent.run(input_prompt, message_history=self.interaction_agent_history.all_messages())
            else:
                result = await self.interactor_agent.run(input_prompt)

            self.interaction_agent_history = result

        except Exception as e:
            print(f"Failed to run interactor agent. {e}")
            await self.manage_tasks(self.task_list, self.task_def,task_running=self.task_running, status = cl.TaskStatus.FAILED , revision_counter = None)
            time.sleep(1)
        
        if result is None:
            result = await self.interactor_agent.run("Inform the user that something went wrong on the server side and ask to try again.")
            await self.manage_tasks(self.task_list, self.task_def,task_running="Thinking", status = cl.TaskStatus.DONE , revision_counter = None)
            #return result.data

        await self.manage_tasks(self.task_list, self.task_def,task_running="Thinking", status = cl.TaskStatus.DONE , revision_counter = self.final_output_revision_counter)
        
        
        return result   
    
    async def generate_response(self, input: str, history : list):
        """
        Generate a response to the user's input using the interactor agent.

        Parameters
        ----------
        input : str
            The user's input to be processed by the interactor agent.
        history : list
            The conversation history to be used as a context for generating the response.

        Returns
        -------
        str
            The generated response to the user's input.
        """
        
        input_prompt = f"""Here you have the most recent conversation history to be used as a context : {history}
        Here you have the user input: {input}
        """
        refinement_feedback = None
        result = None
        retry = 5
        
        for i in range(retry): 
            try: 
                result = await self.call_interactor_agent(input_prompt, result, refinement_feedback)
                
                print("\n-----------------------")
                print("Agent output : ",result.data)
                print("--------------------------")

                break
            except Exception as e:
                print("Failed to provide the final output : ",e)
                if i < retry-1:
                    print(f"Retry counter {i} - Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    break 

        if i == retry -1:
            await self.manage_tasks(self.task_list, self.task_def,task_running="Thinking", status = cl.TaskStatus.RUNNING , revision_counter = self.final_output_revision_counter)
            result = await self.call_interactor_agent("Inform the user that something went wrong on the server side and ask to try again.", result, refinement_feedback)
            await self.manage_tasks(self.task_list, self.task_def,task_running="Thinking", status = cl.TaskStatus.DONE , revision_counter = None)

        return result.data
    

    async def manage_tasks(self, task_list, task_def, task_running, status, revision_counter = None):
        """
        This function is used to manage the task status in the UI. When a task is running, it changes the status of the task to 'Running...'. When the task is finished, it changes the status of the task to 'Done'.

        It also updates the title of the task if the task has a revision_counter greater than 1. The title is updated to include the revision number.

        Parameters:
        task_list (TaskList): The list of tasks.
        task_def (dict): A dictionary containing the task definitions.
        task_running (str): The name of the task that is currently running.
        status (str): The status of the task.
        revision_counter (int): The revision number of the task. If it is greater than 1, it updates the title of the task to include the revision number.
        """
        task_json = task_def[task_running]
        self.task_running = task_running

        if task_list is not None and len(task_list.tasks) > 0:
            
            task_id = list(task_json)[0]
            task_list.tasks[task_id].status = status

            if revision_counter is not None and revision_counter >1: 
                task_list.tasks[task_id].title = task_json[task_id] + f" (Revision {revision_counter})"

        await task_list.send()
        await  cl.sleep(1)
                

