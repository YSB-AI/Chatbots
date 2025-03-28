import chainlit as cl
import copy
import os
from llama_index.core.postprocessor import LLMRerank
import ollama
from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from prompts import *
from ollama import AsyncClient, Client
import re
from fpdf import FPDF
from langgraph.graph import MessagesState, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import  AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from tools_definition import *
import operator
import time
import unicodedata
import json
host = os.getenv("host", None)
user = os.getenv("user", None)
password = os.getenv("password", None)
port = os.getenv("port", None)
database= os.getenv("database", None)
table_name= os.getenv("table_name", None)
decision_agent_model = os.getenv("decision_agent_model", None)
tool_agent_model = os.getenv("tool_agent_model", None)
llama_model_id = os.getenv("llama_model_id", None)


intent_options = ["NORMAL", "DOCUMENTS"]

class UserIntentModel(BaseModel):
    output: Literal[*intent_options]
    rationale : str

class CriticModel(BaseModel):
    observation : Dict
    score : int

class DocSectionsModel(BaseModel):
    title : str
    section_objective : str
    subsections: dict  

class DocsPlanningModel(BaseModel):
    doc_title : str
    doc_objective : str
    sections: List[DocSectionsModel]

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    task : str
    intent : dict
    deep_intent : dict
    intent_history : list
    intent_critic: dict
    
    topics : str
    innovation: str
    innovation_critic : str
    innovation_history : list

    intent_revision_number: int
    intent_max_revisions: int

    innovation_revision_number: int
    innovation_max_revisions: int

    doc_planning : str
    doc_details : dict

    doc_supervisor_history : list
    doc_supervisor_outputs: dict
    docs_written_sections : list

    task_list: cl.TaskList
    task_def : dict

    sources : list
    search_result : str

    sections_review_control : dict
    sections_max_review : int

    final_doc : str
    doc_title : str


class AIAgent():
    def __init__(
            self,
            retriever = None,
            checkpointer = None,
            stream_thread = None,
            **kwargs
            ):

        self.reranker =  LLMRerank(choice_batch_size=20, top_n=20)
        self.extra_params = kwargs
        self.info_provided = False
        self.innovative_works_cache = None

        graph = StateGraph(AgentState)
        graph.add_node("intent_node", self.get_intent_node)
        graph.add_node("intent_deeper_node", self.intent_deeper_node)
        graph.add_conditional_edges("intent_node", self.validate_intent_decision, {
                True : "intent_deeper_node",
                False: "conversational_node"
                }
            )

        graph.add_node("intent_critic_node", self.intent_critic_node)
        graph.add_edge("intent_deeper_node", "intent_critic_node")
        graph.add_node("conversational_node", self.conversational_node)
        graph.add_edge("conversational_node", END)
        graph.add_node("innovation_creator_node", self.innovation_creator_node)
        graph.add_node("innovation_creator_critic_node", self.innovation_creator_critic_node)
        graph.add_edge("innovation_creator_node", "innovation_creator_critic_node")
        graph.add_node("doc_creation_planning_node", self.doc_creation_planning_node)
        graph.add_node("doc_supervisor_node", self.doc_supervisor_node)
        graph.add_edge("doc_creation_planning_node", "doc_supervisor_node")
        graph.add_node("doc_writer_node", self.doc_writer_node)
        graph.add_edge("doc_writer_node", "doc_supervisor_node")

        graph.add_conditional_edges("doc_supervisor_node", self.evaluate_doc_content, {
                True : "doc_writer_node",
                False: "conversational_node"
                }
            )

        graph.add_conditional_edges("intent_critic_node", self.validate_deeper_intent_critic_decision, {
                "NORMAL" : "conversational_node",
                "MISSING_TOPIC" : "conversational_node",
                "SEARCH_INNOVATION" : "innovation_creator_node",
                "DOC_WRITER" : "doc_creation_planning_node",
                False: "intent_deeper_node"
                }
            )

        graph.add_conditional_edges("innovation_creator_critic_node", self.validate_innovation_critic_decision, {
                True : "conversational_node",
                False: "innovation_creator_node"
                }
            )
        graph.set_entry_point("intent_node")

        self.graph = graph.compile(checkpointer=checkpointer)
        self.agent = Client()
        self.innovation_agent = Client()
        self.avaluator_agent = Client()
        self.tools_definition = ToolsDefinition()
        self.retriever = retriever
        self.thread = stream_thread

    def manage_tasks(self, task_list, task_def, task_running, status, revision_counter = None):

        task_json = task_def[task_running]
        
        if task_list is not None and len(task_list.tasks) > 0:
            
            task_id = list(task_json)[0]
            task_list.tasks[task_id].status = status
    
            if revision_counter is not None and revision_counter >1: 
                task_list.tasks[task_id].title = task_json[task_id] + f" (Revision {revision_counter})"

    
    def get_intent_node(self, state: AgentState):
        user_input = state['task']
        conversation_history =  []
        most_recent_conversation_history  = cl.user_session.get("most_recent_history")
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)

        self.manage_tasks(task_list, task_def, task_running="intent_node",  status = cl.TaskStatus.RUNNING,  )

        print("\n----------------------------")
        print(f"Getting intent for user's input : {user_input}")
        print(f"Most recent conversation history : {most_recent_conversation_history}")
        print(f"Conversation history : {conversation_history}")
        print("----------------------------")

        if len(conversation_history) == 0:
            conversation_history.append({"role":"system", "content" : INTENT_PROMPT})
            
        prompt = f"""
        Here you have the conversation history with the user : {most_recent_conversation_history}
        Here you have the user's input: {user_input}
        Analyze the inputs carefully and classify the intent.
        """

        conversation_history.append({"role":"user", "content" : prompt})
        for _ in range(3): #Sometimes the ollama model output is broken , that's why we have this loop
            response = self.agent.chat(
                messages=conversation_history,
                model=decision_agent_model,
                format=UserIntentModel.model_json_schema(),
                options = {
                        "temperature": 0.3,
                        "num_predict" : 500
                    }
            )
            try:
                user_intent = UserIntentModel.model_validate_json(response.message.content)
                break
            except Exception as e:
                print(f"Failed to format intent output. ",e)


        self.manage_tasks(task_list, task_def, task_running="intent_node",  status = cl.TaskStatus.DONE)

        return {'intent':  user_intent, 
            "task_list" : task_list,
            "task_def" : task_def,
            }

    def intent_deeper_node(self, state: AgentState):
        
        user_input  = state.get("task", None)
        conversation_history = state.get("intent_history",[])
        intent_critic = state.get('intent_critic', {})
        task_list= state.get("task_list", None)
        most_recent_conversation_history  = cl.user_session.get("most_recent_history")
        task_def= state.get("task_def",None)

        self.manage_tasks(task_list, task_def,task_running="intent_deeper_node", status = cl.TaskStatus.RUNNING , revision_counter = state.get("intent_revision_number"))

        print("\n----------------------------")
        print(f"Getting intent deeper analysis for user's input : {user_input}")
        print("----------------------------")
        
        for phase in INTENT_VERIFICATION_ORDER:
            print("Running deeper intent classification phase : ",phase)
            prompt = copy.deepcopy(INTENT_PHASES_PROMPT[phase]["prompt"]).replace("$CONVERSATION_HISTORY$",str(most_recent_conversation_history)).replace("$USER_INPUT$", str(user_input)).replace("$INTENT_CRITIQUE$", str(intent_critic))
            intent_opt = INTENT_PHASES_PROMPT[phase]["output"]
            
            conversation_history.append({"role":"user", "content" : prompt})
        
            class IntentResearchCheckModel(BaseModel):
                output: Literal[*intent_opt]
                rationale : str

            for _ in range(3): #Sometimes the ollama model output is broken , that's why we have this loop
                try:
                        
                    response = self.avaluator_agent.chat(
                        messages=conversation_history,
                        model=decision_agent_model,
                        format=IntentResearchCheckModel.model_json_schema(),
                        options = {
                            "temperature": 0.3,
                            "num_predict" : 500
                        }
                    )
                    deep_intent = IntentResearchCheckModel.model_validate_json(response.message.content)
                    print(f"Deeper inten analysis phase '{phase}' output -> ",deep_intent)

                    break
                except Exception as e:
                    print("Error validating intent critic decision: ",e) 


            if deep_intent.output == "NO_LIST_OF_WORKS":
                conversation_history.append({'role': 'assistant', 'content':  "There is not a list of innovative works in the conversation history."})

            elif deep_intent.output == "LIST_EXISTS":
                conversation_history.append({'role': 'assistant', 'content':  "There is a list of innovative works in the conversation history."})  

            elif deep_intent.output == "NOT_SELECTING_OPTION":
                conversation_history.append({'role': 'assistant', 'content':  "The user is NOT selecting a work/option from a provided list of innovative works (provided by the assistant)."})
            else:
                break

        conversation_history.append({'role': 'assistant', 'content':  f"**Intent classifier** output : {str(deep_intent)} "})

        print("Last deeper analysis on intent critic decision: ",deep_intent)

        self.manage_tasks(task_list, task_def, task_running="intent_deeper_node", status = cl.TaskStatus.DONE, revision_counter = state.get("intent_revision_number"))
        return {
            "deep_intent" : deep_intent,
            "intent_revision_number": state.get("intent_revision_number", 1) + 1,
            "intent_history" : conversation_history,
            "task_list" : task_list,
            "task_def" : task_def
        }


    def intent_critic_node(self, state: AgentState):
        print("Getting intent critics for intent deeper analysis ...")
        
        conversation_history = state.get("intent_history",[])
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)

        self.manage_tasks(task_list, task_def,task_running="intent_critic_node", status = cl.TaskStatus.RUNNING)


        for _ in range(3): #Sometimes the ollama model output is broken , that's why we have this loop
            try:
               
                prompt = f"""{INTENT_CRITIC_PROMPT}
                Now, perform an in-depth evaluation and provide your critique.
                """
                conversation_history.append({"role":"user", "content" : prompt})

                response = self.avaluator_agent.chat(
                    messages=conversation_history,
                    model=decision_agent_model,
                    format=CriticModel.model_json_schema(),
                    options = {
                        "temperature": 0.5,
                        "num_predict" : 500
                    }
                )
            
                intent_critic = CriticModel.model_validate_json(response.message.content)
                break
            except Exception as e:
                print(f"Error validating intent critic decision :{response.message.content}. \n{e}") 

        self.manage_tasks(task_list, task_def, task_running="intent_critic_node", status = cl.TaskStatus.DONE)
        conversation_history.append({'role': 'assistant', 'content':  f"**Critiques** for **Intent classifier** : {intent_critic.observation} "})

        return {"intent_critic" : {
            "score" : intent_critic.score,
            "observation" : intent_critic.observation
        }, 
            "intent_history" : conversation_history,
            "task_list" : task_list,
            "task_def" : task_def
        }

    def validate_intent_decision(self, state: AgentState):

        intent = state.get("intent", None)
        print("\n----------------------------")
        print(f"Validating intent decision for intent analysis : {intent}")
        print("----------------------------")
        
        if intent is None :
            raise Exception("Missing intent ..")

        if state["intent_revision_number"] > state["intent_max_revisions"]:
            print("Maximum revision reached...")
            return "NORMAL"
    
        if intent.output == "NORMAL":
            return False  
        else:
            return True

    def validate_deeper_intent_critic_decision(self, state: AgentState):

        intent_critic_decision = state.get("intent_critic", {})
        deep_intent = state.get("deep_intent", None)

        print("\n----------------------------")
        print(f"Validating the decision for deeper intent critic analysis : {intent_critic_decision}")
        print("----------------------------")

        if deep_intent is None :
            raise Exception("Missing intent ..")


        if state["intent_revision_number"] > state["intent_max_revisions"]:
            print("Maximum revision reached...")
            return "NORMAL"
            
        score = float(intent_critic_decision.get("score", 0))

        if score >= 85:  # HIGH CONFIDENCE - Accept the classification as correct
            print(f"High agreement ({score}%), proceeding with deeper intent:", deep_intent.output)
            
            # **Special case: DOC_WRITER requires innovation selection**
            # if deep_intent.output == "DOC_WRITER" and not state.get("innovation"):
            #     print("DOC_WRITER intent detected, but no innovation was selected. Rejecting.")
            #     return False  # Force revision

            return deep_intent.output

        elif 50 <= score < 85:  # MEDIUM CONFIDENCE - Could be correct, but needs more checks
            print(f"Medium agreement ({score}%). Checking if revisions are needed...")

            # **Handle borderline cases: Accept with constraints**
            if (cl.user_session.get("resumed_conversation") == False and deep_intent.output in ["SEARCH_INNOVATION"] and not state.get("innovation")) or (cl.user_session.get("resumed_conversation") == False and deep_intent.output in ["DOC_WRITER"] and state.get("innovation")) or (cl.user_session.get("resumed_conversation") == True):
                print("Intent involves innovation, allowing progression.")
                return deep_intent.output

            print("Unclear decision, requesting revision...")
            return False  # Request another review

        else:  #  LOW CONFIDENCE - Model is unsure, revision is needed
            print(f"Low agreement ({score}%), rejecting deeper intent. Reverting to intent_node...")
            return False  # Force re-evaluation


    def conversational_node(self, state: AgentState):
        print( "Conversational node...")

        intent  = state.get("intent", None)
        deep_intent = state.get("deep_intent", None)
        user_input  = state.get("task", None)
        message_history  = state.get("messages", None)
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        source= state.get("sources", [])
        final_doc = state.get("final_doc", {})

        self.manage_tasks(task_list, task_def,task_running="conversational_node", status = cl.TaskStatus.RUNNING)


        if intent is None :
            raise Exception("Missing intent ..")
        elif  user_input is None:
            raise Exception("Missing user input ..")

        prompt = f"""You are a conversational agent that will interact with a user. You will receive a users's input, the conversation history (if any) and an instruction .
        Your task is to follow strictly the instruction.
        Here you have the user's input : {user_input}",
        """
        
        if deep_intent is not None:
            if deep_intent.output =="MISSING_TOPIC" or deep_intent.output.find('MISSING_TOPIC') != -1:
                prompt = prompt + f"\nThe user wants to create a document but the topic is missing. Ask explicitly for the topic of the document to be created. Be concise"

            elif deep_intent.output =="SEARCH_INNOVATION" or deep_intent.output.find('SEARCH_INNOVATION') != -1:
                innovation = state.get("innovation", None)
                prompt = f"""Present the following innovative works ans ask the user to select one of them : {innovation}."""
            
            elif deep_intent.output =="DOC_WRITER" or deep_intent.output.find('DOC_WRITER') != -1:
                prompt = f"""Summarize the created work: {final_doc}."""

            elif intent.output == "NORMAL" or intent.output.find('NORMAL') != -1:
                prompt = prompt +f"\n Inform the user that you are an agent specifialized in guidance for document creation. Be concise."
            
            else:
                prompt = prompt +"\nAsk the user for refinement of its input as it was not clear for you."
        else:
            if intent.output == "NORMAL" or intent.output.find('NORMAL') != -1:
                prompt = prompt +f"\n Inform the user that you are an agent specifialized in guidance for document creation. Be concise."
            
            else:
                prompt = prompt +"\nAsk the user for refinement of its input as it was not clear for you."

        prompt = prompt + f"\nOutput only the response without your any extra comments or explanation and using the language {cl.user_session.get('language')}"
        
        message_history.append({"role":"user", "content" : prompt})

        response = self.agent.chat(
            messages=message_history,
            model=llama_model_id,
            
        )

        self.manage_tasks(task_list, task_def,task_running="conversational_node", status = cl.TaskStatus.DONE)

        return {'messages': [{
                    'role': 'assistant',
                    'content': response.message.content
                }],
            "task_list" : task_list,
            "task_def" : task_def  ,
            "sources" : source}
    

    def retrieve_chunks(self,  query : str):
        
        sources =  []
        ids_retrieved = []
        sources_elements = []

        query_result_list = self.retriever.retrieve(query)

        if len(query_result_list)>0:
            for result in query_result_list:
                chunk_id = result.node.id_
                metadata= result.node.metadata

                if result.score > 0.3: 
                    if  chunk_id not in ids_retrieved:
                        ids_retrieved.append(chunk_id)
                        sources.append(result.node.text)
                        # Sources to be presented
                        doc_name = metadata["file_path"].split("/")[-1].replace(".pdf","")
                        sources_elements.append(cl.Pdf(name=doc_name, path = metadata["file_path"], display="inline") )

        str_sources = '\n'.join(sources)
        return str_sources, sources_elements, sources


    def innovation_creator_node(self, state: AgentState):
        print("\nInnovationg creator node ...")

        user_input  = state.get("task", None)
        conversation_history  = state.get("innovation_history", [])
        innovation_critic = state.get("innovation_critic", None)
        innovation_revision_number = state.get("innovation_revision_number", None)
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        search_result = state.get("search_result",None)


        self.manage_tasks(task_list, task_def,task_running="innovation_creator_node", status = cl.TaskStatus.RUNNING, revision_counter=  innovation_revision_number)
  
        tools_list = [
            self.tools_definition.retrieve_data_olama_tool(), 
            ]
        if len(conversation_history) == 0 :
            conversation_history.append({"role":"system", "content" : INNOVATION_SYSTEM_PROMPT})
     
        inputs = f"""
        - User input : {user_input}
        - **Consider the user's critique and evaluation** (`{innovation_critic}`) to refine your output.
        - **Use the language:** `{cl.user_session.get('language')}` for the final output.

        """

        conversation_history.append({"role":"user", "content" : inputs})

        for i in range(30):#Sometimes the ollama model output is broken , that's why we have this loop

            response = self.innovation_agent.chat(
                    model= llama_model_id ,
                    messages= conversation_history,
                    tools = tools_list,
                    options = {
                        "temperature": cl.user_session.get("temperature_param"),
                        "num_predict" : 8000,
                    },
                    
                )

            output = ""
            topics = None
            
            print("List of tools : ", response.message.tool_calls)
            str_sources=""
            sources =  []
            sources_elements = state.get("sources", [])

            if not response.message.tool_calls :
                print("Retrying innovation because no retrieval was triggered.")

                if i ==0:
                    conversation_history.append(response.message)
                    conversation_history.append({"role":"user", "content" : "You FAILED to call the tool 'retrieve_data_tool'. **This is mantadory as you need to call it to get data for innovation creation.** Make sure to call the tool. "})
            else:
                break

        while response.message.tool_calls:

            for tool in response.message.tool_calls:
                
                print(f"Running tool : {tool.function.name} with parameters {tool.function.arguments}")

                if tool.function.name == "retrieve_data_tool":
                    print("Retrieving data ...")

                    if len(tool.function.arguments) == 1 :
                        k = list(tool.function.arguments)[0]
                        topics = str(tool.function.arguments[k])
                        
                        if topics is None:
                            output = f" You have selected the tool {tool.function.name} but no valid topics were extracted. Focus on extracting valid topics."
                        else:
                            str_sources, sources_elements, sources = self.retrieve_chunks(topics)

                            print(f"Finished retriving chunks with {len(sources_elements)} results...")

                            if len(sources_elements)> 0 :
                                output = self.get_innovation_works(topics, sources, innovation_critic, str_sources)
                            else:
                                output = f"No relevant data found and no innovations were proposed for the topics '{topics}'.",

                    else:
                        output = f" You have selected the tool {tool.function.name} but no valid topics were extracted. Focus on extracting valid topics."

                                      
                print("Calling the agent again")
                conversation_history.append(response.message)
                conversation_history.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
            
            response = self.innovation_agent.chat(
                decision_agent_model, 
                messages=conversation_history,
                tools = tools_list
                )

        conversation_history.append({'role': 'assistant', 'content':  f" Output :  {response.message.content}"})
        self.manage_tasks(task_list, task_def, task_running="innovation_creator_node", status = cl.TaskStatus.DONE)

        print("Identified topics : ", topics)
        print("----------------------------")
        return {
            "innovation": response.message.content, 
            "innovation_revision_number": state.get("innovation_revision_number", 1) + 1,
            "topics" : topics,
            "innovation_history" : conversation_history,
            "task_list" : task_list,
            "task_def" : task_def,
            "sources" : sources_elements,
            "search_result" : str_sources
        }

    def get_innovation_works(self, topics, sources, innovation_critic, str_sources):

        innovation_prompt = copy.deepcopy(INNOVATION_TOOL_PROMPT)
        innovation_prompt = innovation_prompt.replace("$TOPICS$", str(topics)).replace("$SEARCH_RESULTS$", "\n".join(sources)).replace("$CRITIC_CONVERSATION_HISTORY$", str(innovation_critic))
        response = self.innovation_agent.chat(
                model=decision_agent_model,#tool_agent_model,
                messages= [{"role":"user", "content" : innovation_prompt}],
                options = {
                    "temperature": cl.user_session.get("temperature_param"),
                    "num_predict" : 8000
                }
            )

        output = f"""Here you have the topics : {str(topics)}
        Here you have the search result : {str_sources}
        Here you have a set of innovative works (title and description) : {str(response.message.content)}
        """
        return output

    def innovation_creator_critic_node(self, state: AgentState):
        
        innovation = state.get("innovation", None)
        conversation_history  = state.get("innovation_history", [])
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        search_result = state.get("search_result",None)

        self.manage_tasks(task_list, task_def,task_running="innovation_creator_critic_node", status = cl.TaskStatus.RUNNING)

        print("\n----------------------------")
        print(f"Getting innovation critics for : {innovation}")
        print("----------------------------")

        if innovation is None :
            raise Exception("Missing innovation ..")
        
    
        prompt = INNOVATION_CRITIC_PROMPT.replace("$SEARCH_RESULT$",search_result)

        conversation_history.append({'role': 'user','content': prompt })

        response = self.avaluator_agent.chat(
            messages=conversation_history,
            model=decision_agent_model, #tool_agent_model,
            format=CriticModel.model_json_schema(),
        )

        innovative_works_critics = CriticModel.model_validate_json(response.message.content)

        conversation_history.append({'role': 'assistant', 'content':  f"Critics for the innovative works and its creation process : {str(innovative_works_critics.observation)} "})

        self.manage_tasks(task_list, task_def, task_running="innovation_creator_critic_node", status = cl.TaskStatus.DONE)

        return {"innovation_critic" : {
            "observation" : innovative_works_critics.observation,
            "score" : innovative_works_critics.score
        }, 
            "innovation_history" :conversation_history,
            "task_list" : task_list,
            "task_def" : task_def
        }
       
    def validate_innovation_critic_decision(self, state: AgentState):
        
        innovation_critic_decision = state.get("innovation_critic", {})
        print("\n----------------------------")
        print(f"Validating innovation critic decision : {innovation_critic_decision}")
        print("----------------------------")
    
        if float(innovation_critic_decision["score"]) >= 70.0 or state["innovation_revision_number"] > state["innovation_max_revisions"]:
            return True
            
        else:
            return False
    
    def doc_creation_planning_node(self, state: AgentState):
        print("Document creation planning node ...")

        messages  = state.get("messages", [])
        user_input  = state.get("task", None)
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        planning= state.get("planning",{})


        self.manage_tasks(task_list, task_def,task_running="doc_creation_planning_node", status = cl.TaskStatus.RUNNING)
        
        prompt = DOC_PLANNING_PROMPT.replace("$CONVERSATION_HISTORY$", str(messages)).replace("$USER_INPUT$", str(user_input)) 

        for _ in range(3):#Sometimes the ollama model output is broken , that's why we have this loop
            try:
                response = self.agent.chat(
                    messages=[{
                    'role': 'user',
                    'content': prompt,
                    }],
                    model=decision_agent_model,
                    format=DocsPlanningModel.model_json_schema(),
                )
                planning_formated_output = DocsPlanningModel.model_validate_json(response.message.content)
                break
            except Exception as e:
                    print("Planning node response : ",response.message.content)
                    print(f"""Error in doc_creation_planning_node : {e}""")


        planning = []
        for i ,section in enumerate(planning_formated_output.sections):
            planning.append({
                "title" : section.title,
                "objective" : section.section_objective,
                "subsections" : section.subsections
            })
        print("Document selected : ", planning_formated_output.doc_title)
        print("Document selected objectives : ", planning_formated_output.doc_objective)
        print("Plan sections formatted -> ", planning)
    
        print("Updating the search result for this work  ...")
        str_sources, sources_elements, sources = self.retrieve_chunks(planning_formated_output.doc_title+" : "+ planning_formated_output.doc_objective)

        self.manage_tasks(task_list, task_def, task_running="doc_creation_planning_node", status = cl.TaskStatus.DONE)

        return {
            "messages" : [
                {
                "role": "assistant",
                "content": str(response.message.content)
                }
            ],
            "doc_planning" : response.message.content,
            "task_list" : task_list,
            "task_def" : task_def,
            "doc_details": {"planning": planning,
                            "doc_title" : planning_formated_output.doc_title,
                            "doc_objective" : planning_formated_output.doc_objective
            },
            "sources" : sources_elements,
            "search_result" : str_sources
        }

    def doc_supervisor_node(self, state: AgentState):
        print("Document supervisor node ...")

        doc_planning  = state.get("doc_planning", None)
        doc_supervisor_history  = state.get("doc_supervisor_history", [])
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        doc_details = state.get("doc_details", {})
        sections_review_control = state.get("sections_review_control", {})
        sections_max_review = state.get("sections_max_review", 5)
        doc_supervisor_outputs = state.get("doc_supervisor_outputs", {})
        final_doc = state.get("final_doc", "")

        self.manage_tasks(task_list, task_def,task_running="doc_supervisor_node", status = cl.TaskStatus.RUNNING)

        if len(doc_supervisor_history) == 0:
            system_prompt = copy.deepcopy(DOC_SUPERVISOR_PROMPT)
            system_prompt = system_prompt.replace("$DOC_PLANNING$", str(doc_planning)).replace("$DOC_TITLE$",doc_details["doc_title"]).replace("$DOC_OBJECTIVE$",doc_details["doc_objective"])
            doc_supervisor_history.append({"role":"system", "content" : system_prompt})


        if 'next_section' in doc_supervisor_outputs:
            prompt = """
            Criticize and evaluate the latest sections written according to their defined objectives in the plan. If any section does not meet the objectives, select the SAME section again for refinement and add an 1-2 sentences observation about what could be improved. Be specific in your observation, pointing out the issues. 
            If no critical improvements needed, select the next section to be written and add its respective objectives according to the provided plan.
            """
            
            last_section_id = doc_supervisor_outputs['next_section']
            print(f"The supervisor is evaluating the {last_section_id} and deciding what to do next...")
            prompt = prompt + f"\n\n Here you have the last section '{last_section_id}' results : {doc_supervisor_outputs[last_section_id]['results']}"
        else : 
            prompt = """
            Select the next section to be written and add its respective objectives according to the provided plan.
            """

        doc_supervisor_history.append({"role":"user", "content" : prompt})

        available_sections = ["FINISH"]
        for details in doc_details["planning"]:
            if details["title"] in sections_review_control:
                try:
                    print(f" {details['title']} -> {sections_review_control[details['title']]}/{sections_max_review}")
                    if sections_review_control[details["title"]] < sections_max_review:
                        available_sections.append(details["title"])
                    # else:
                    #     final_doc = final_doc + doc_supervisor_outputs[details["title"]]['results']+"\n\n"

                except Exception as e:
                    print(f"Failed to append/evaluate {details['title']} final doc : {e}")
                    print("doc_supervisor_outputs : ",doc_supervisor_outputs)

            else:
                available_sections.append(details["title"])

        print("Available sections : ",available_sections)

        class Router(BaseModel):
            next_section: Literal[*available_sections]
            objectives : str
            subsections : str
            observation : str

        for _ in range(3):#Sometimes the ollama model output is broken , that's why we have this loop
            try:
                response =  self.agent.chat(
                        model=decision_agent_model,
                        messages= doc_supervisor_history,
                        format=Router.model_json_schema(),
                    )
                router = Router.model_validate_json(response.message.content)

            except Exception as e:
                print(f"""Error in doc_supervisor_node : {e}""")

        output_content = f"""
        Next section to be written : {router.next_section}
        Objectives of the section : {router.objectives}
        Subsections and their objectives : {router.subsections}
        Observations for refinement : {router.observation}
        """

        doc_supervisor_history.append({"role": "assistant", "content":  output_content})

        if "next_section" in doc_supervisor_outputs:
            if last_section_id in doc_supervisor_outputs:
                doc_supervisor_outputs[last_section_id]['observation'] = router.observation
                print(f"Supervisor last section  observation '{last_section_id}' : {router.observation}")

        doc_supervisor_outputs["next_section"] = router.next_section
        doc_supervisor_outputs["objectives"] = router.objectives
        doc_supervisor_outputs["subsections"] = router.subsections

        print(f"Supervisor output : '{router.next_section}' with objectives '{router.objectives}' ")

        if router.next_section == "FINISH":
            for details in doc_details["planning"]:
                if details["title"] in doc_supervisor_outputs:
                    final_doc = final_doc + doc_supervisor_outputs[details["title"]]['results']+"\n\n"

        self.manage_tasks(task_list, task_def, task_running="doc_supervisor_node", status = cl.TaskStatus.DONE)

        return {
            "doc_supervisor_history" : doc_supervisor_history,
            "doc_supervisor_outputs" : doc_supervisor_outputs,
            "docs_written_sections" : state.get("docs_written_sections", []),
            "task_list" : task_list,
            "task_def" : task_def,
            "doc_details" : doc_details,
            "final_doc" : final_doc
        }

    def evaluate_doc_content(self, state: AgentState):
        
        doc_supervisor_outputs = state.get("doc_supervisor_outputs", {})

        if "next_section" in doc_supervisor_outputs:
            if doc_supervisor_outputs["next_section"] == "FINISH":
                print("Document creation finished")

                self.graph.update_state(self.thread, {
                    "innovation_revision_number": 1,
                    "intent_revision_number": 1,
                    "innovation_history" : [],
                    "intent_history" : [],
                    "sources" : [],
                    "search_result" : "",
                    "doc_planning" : "",
                    "doc_supervisor_outputs" : {},
                    "doc_supervisor_history" : [],
                    "docs_written_sections" : [],
                    "sections_review_control" : {},
                    "final_doc" : "",
                    "doc_details" : {},
                    "topics" : ""
                    })
    
                return False
        
        return True

        
    def doc_writer_node(self, state: AgentState):
        print("Doc writer node ...")
        doc_supervisor_outputs = state.get("doc_supervisor_outputs", {})
        docs_written_sections = state.get("docs_written_sections", [])
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        search_result = state.get("search_result",None)
        sections_review_control = state.get("sections_review_control", {})

        self.manage_tasks(task_list, task_def,task_running="doc_writer_node", status = cl.TaskStatus.RUNNING)

        section_id  = doc_supervisor_outputs['next_section']
        doc_writer_history = section_id+"_history"

        if doc_writer_history not in doc_supervisor_outputs:

            system_prompt = copy.deepcopy(DOC_WRITER_SYSTEM_PROMPT).replace("$SECTION_ID$", str(section_id)).replace("$SECTION_OBJECTIVES$", str(doc_supervisor_outputs['objectives'])).replace("$SUBSECTION_OBJECTIVES$", str(doc_supervisor_outputs['subsections'])).replace("$SEARCH_RESULT$", str(search_result))
            doc_supervisor_outputs[doc_writer_history] = [{"role": "system", "content":  system_prompt}]

        if section_id in  docs_written_sections:
            prompt = f"""
            Here you have the critiques to be used for refinement: {doc_supervisor_outputs[section_id]['observation']}
            Now, generate the best section as possible considering its objectives and observations (if any).

            Format the every section or subsection title to be capital letter to enphasize that it is a beginning of a section&subsection.
            Output only the response without your any extra comments or explanation. 
            """
        else:
            prompt = """Now, generate the best section as possible considering its objectives and observations (if any).
            Output only the response without your any extra comments or explanation. """

        # Let's keep the writer history apart from the supervisor's history so that the conversation history will not become too long
        doc_supervisor_outputs[doc_writer_history].append({"role": "user", "content":  prompt})

        response =  self.agent.chat(
                model=decision_agent_model,
                messages= doc_supervisor_outputs[doc_writer_history]
            )

        print("\n----------------------------")
        print(f"{section_id} Output :  {response.message.content}")
        print("----------------------------")
        
        # Update the conversation history for the given section
        doc_supervisor_outputs[doc_writer_history].append({"role": "assistant", "content":  response.message.content})

        # Update  with the last output for the given section
        doc_supervisor_outputs[section_id] = {"results" : response.message.content, "observation" : ""}

        # Sections already written at least once
        if section_id not in docs_written_sections:
            docs_written_sections.append(section_id)

        # Update sections_review_control to keep track of the maximum amount of review for each section
        if section_id in sections_review_control:
            sections_review_control[section_id] = sections_review_control[section_id] +1
        else:
            sections_review_control[section_id] = 1

        self.manage_tasks(task_list, task_def, task_running="doc_writer_node", status = cl.TaskStatus.DONE)
        return {
            "doc_supervisor_outputs" : doc_supervisor_outputs,
            "docs_written_sections" : docs_written_sections,
            "task_list" : task_list,
            "task_def" : task_def,
            "sections_review_control" : sections_review_control
        }


#=== Receives a List of chainlit elements and appends it
def sources_update(elements):
    answer_sources = cl.user_session.get("answer_sources")
    answer_sources.append(elements)
    cl.user_session.set("answer_sources", answer_sources)

#=== Receives a Chainlit Message and appends it
def update_conversation_history(msg):

    conversation_history = cl.user_session.get("conversation_history")
    conversation_history.append(msg)
    cl.user_session.set("conversation_history", conversation_history)

    resume_history = cl.user_session.get("resume_history")
    resume_history.append({msg.author : msg.content})
    cl.user_session.set("resume_history", resume_history)


