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
from graph import *

host = os.getenv("host", None)
user = os.getenv("user", None)
password = os.getenv("password", None)
port = os.getenv("port", None)
database= os.getenv("database", None)
table_name= os.getenv("table_name", None)
decision_agent_model = os.getenv("decision_agent_model", None)
tool_agent_model = os.getenv("tool_agent_model", None)

class OllamaClient():
    def __init__(self, model, options):
        self.model = model
        self.client = ollama.Client()
        self.options = options

    def chat(
        self, 
        messages, 
        format=None, 
        options=None, tools = None
        ):
        
        if options is None:
            options = self.options

        if tools is None:
            tools = []
  
        format_output = None

        for _ in range(3): #Sometimes the ollama model output is broken , that's why we have this loop
            try:
                if format is None:
                
                    response = self.client.chat(
                        model=self.model,
                        messages=messages,
                        options=options,
                        tools = tools
                    )
                else:
                    response = self.client.chat(
                        model=self.model,
                        messages=messages,
                        format=format.model_json_schema(),
                        options=options,
                        tools = tools
                    )

                    format_output = format.model_validate_json(response.message.content)

                return response, format_output
            except Exception as e:
                print(f"Ollama API error: {e}")
                 # Or raise a custom exception

class AIAgent():
    def __init__(
            self,
            retriever = None,
            checkpointer = None,
            stream_thread = None,
            **kwargs
            ):

            
        self.reranker =  LLMRerank(choice_batch_size=20, top_n=20)

        graph = StateGraph(AgentState)
        graph.add_node("intent_deeper_node", self.intent_deeper_node)

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
                "NORMAL_CONVERSATION" : "conversational_node",
                "DOCUMENT_REQUEST_WITH_TOPIC" : "innovation_creator_node",
                "SELECT_FROM_OPTIONS" : "doc_creation_planning_node",
                False: "intent_deeper_node"
                }
            )

        graph.add_conditional_edges("innovation_creator_critic_node", self.validate_innovation_critic_decision, {
                True : "conversational_node",
                False: "innovation_creator_node"
                }
            )
        graph.set_entry_point("intent_deeper_node")

        self.graph = graph.compile(checkpointer=checkpointer)
        self.tool_agent= OllamaClient(tool_agent_model, options={"temperature": cl.user_session.get("temperature_param"),"num_predict" : 8000})
        self.agent = OllamaClient(decision_agent_model, options={"temperature": cl.user_session.get("temperature_param"),"num_predict" : 8000})
        self.innovation_agent = OllamaClient(decision_agent_model, options={"temperature": cl.user_session.get("temperature_param"),"num_predict" : 8000})
        self.tools_definition = ToolsDefinition()

        self.retriever = retriever
        self.thread = stream_thread

    
    def intent_deeper_node(self, state: AgentState):
        """
        This function performs a deeper analysis of the user's intent by evaluating various phases 
        of intent classification. It checks for specific criteria in the conversation history and 
        user input to determine the appropriate classification and output.

        Parameters:
        - state (AgentState): An object representing the current state of the agent, containing 
        information such as user input, intent history, intent critic, and task details.

        Returns:
        - dict: A dictionary containing the deep intent classification, updated intent revision 
        number, intent history, and task details.

        The function manages the task status, prints debug information, and handles exceptions 
        during the intent analysis process.
        """

        user_input  = state.get("task", None)
        conversation_history = state.get("intent_history",[])
        intent_critic = state.get('intent_critic', {})
        task_list= state.get("task_list", None)
        most_recent_conversation_history  = cl.user_session.get("most_recent_history")
        task_def= state.get("task_def",None)
        message_history  = state.get("messages", None)

        manage_tasks(task_list, task_def,task_running="intent_deeper_node", status = cl.TaskStatus.RUNNING , revision_counter = state.get("intent_revision_number"))

        print("\n----------------------------")
        print(f"Getting intent deeper analysis for user's input : {user_input}")
        print("----------------------------")
        
        class IntentResearchCheckModel(BaseModel):
            output: Literal[
                "NORMAL_CONVERSATION",
                "DOCUMENT_REQUEST_WITH_TOPIC",
                "SELECT_FROM_OPTIONS",
            ]
            reasoning: str

        prompt = copy.deepcopy(DOCUMENT_INTENT_PROMPT).replace("$CONVERSATION_HISTORY$", str(most_recent_conversation_history)).replace("$USER_INPUT$", str(user_input)).replace("$INTENT_CRITIQUE$", str(intent_critic))

        conversation_history.append({"role":"user", "content" : prompt})
        message_history.append({"role":"user", "content" : prompt})

        options = {
            "temperature": 0.1,
            "num_predict" : 600,
            "top_p" :0.9
        }
        _, deep_intent = self.agent.chat(conversation_history, format=IntentResearchCheckModel, options=options, tools = None)
        print("Deeper intent analysis output -> ",deep_intent)

        conversation_history.append({'role': 'assistant', 'content':  f"**Intent classifier** output : {str(deep_intent)} "})
        message_history.append({'role': 'assistant', 'content':  f"**Intent classifier** output : {str(deep_intent)} "})


        manage_tasks(task_list, task_def, task_running="intent_deeper_node", status = cl.TaskStatus.DONE, revision_counter = state.get("intent_revision_number"))
        return {
            "deep_intent" : deep_intent,
            "intent_revision_number": state.get("intent_revision_number", 1) + 1,
            "intent_history" : conversation_history,
            "task_list" : task_list,
            "task_def" : task_def,
            "messages" : message_history
        }


    def intent_critic_node(self, state: AgentState):
        """
        This function validates the decision for deeper intent critic analysis based on the intent critic score and deeper intent.

        This function assesses whether the deeper intent, as analyzed by the intent critic, should be accepted or 
        requires further revision. It takes into consideration the critic's confidence score, the status of the 
        conversation (resumed or not), and whether the maximum number of intent revisions has been reached.

        Parameters:
        - state (AgentState): An object representing the current state of the agent, containing information 
        such as intent critic decision, deep intent, innovation, and revision counts.

        Raises:
        - Exception: If the deep intent is missing from the state.

        Returns:
        - str: The output of the deep intent if accepted.
        - bool: False if further revision is required.
        """
        print("Getting intent critics for intent deeper analysis ...")
        
        conversation_history = []#state.get("intent_history",[])
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        message_history  = state.get("messages", None)
        user_input  = state.get("task", None)
        deep_intent= state.get("deep_intent", None)
        most_recent_conversation_history  = cl.user_session.get("most_recent_history")

        manage_tasks(task_list, task_def,task_running="intent_critic_node", status = cl.TaskStatus.RUNNING)
        
        prompt = copy.deepcopy(INTENT_CRITIQUE_PROMPT).replace("$CONVERSATION_HISTORY$", str(most_recent_conversation_history)).replace("$USER_INPUT$", str(user_input))
        prompt = prompt + f"""
        Here you have the **Intent classifier** output and reasoning : {str(deep_intent)} 
        Now, perform an in-depth evaluation and provide your critique."""
        conversation_history.append({"role":"user", "content" : prompt})
        message_history.append({"role":"user", "content" : prompt})
        
        options = {

                "temperature": 0.2,
                "num_predict" : 1000,
                "top_p" : 0.9
            }
        _, intent_critic = self.agent.chat(conversation_history, format=CriticModel, options=options, tools = None)

        manage_tasks(task_list, task_def, task_running="intent_critic_node", status = cl.TaskStatus.DONE)
        conversation_history.append({'role': 'assistant', 'content':  f"**Critiques** for **Intent classifier** : {intent_critic} "})
        message_history.append({'role': 'assistant', 'content':  f"**Critiques** for **Intent classifier** : {intent_critic} "})

        return {"intent_critic" : {
            "score" : intent_critic.score,
            "observation" : intent_critic.observation,
            "recommendations" : intent_critic.recommendations
        }, 
            "task_list" : task_list,
            "task_def" : task_def,
            "messages" : message_history
        }


    def validate_deeper_intent_critic_decision(self, state: AgentState):

        """
        Validates the decision for deeper intent critic analysis based on the intent critic score and deeper intent.

        This function assesses whether the deeper intent, as analyzed by the intent critic, should be accepted or 
        requires further revision. It takes into consideration the critic's confidence score, the status of the 
        conversation (resumed or not), and whether the maximum number of intent revisions has been reached.

        Parameters:
        - state (AgentState): An object representing the current state of the agent, containing information 
        such as intent critic decision, deep intent, innovation, and revision counts.

        Raises:
        - Exception: If the deep intent is missing from the state.

        Returns:
        - str: The output of the deep intent if accepted.
        - bool: False if further revision is required.
        """

        intent_critic_decision = state.get("intent_critic", {})
        deep_intent = state.get("deep_intent", None)

        print("\n----------------------------")
        print(f"Validating the decision for deeper intent critic analysis : {intent_critic_decision}")
        print(f"Deep intent : {deep_intent}")
        print("----------------------------")

        if deep_intent is None :
            raise Exception("Missing intent ..")


        if state["intent_revision_number"] > state["intent_max_revisions"]:
            print("Maximum revision reached...")
            return "NORMAL_CONVERSATION"
            
        score = float(intent_critic_decision.get("score", 0))

        if score >= 85.0:  # HIGH CONFIDENCE - Accept the classification as correct
            print(f"High agreement ({score}%), proceeding with deeper intent:", deep_intent.output)

            return deep_intent.output

        elif 50.0 <= score < 74.0:  # MEDIUM CONFIDENCE - Could be correct, but needs more checks
            print(f"Medium agreement ({score}%). Checking if revisions are needed...")

            # **Handle borderline cases: Accept with constraints**
            if  (deep_intent.output in ["NORMAL_CONVERSATION"]) or (cl.user_session.get("resumed_conversation") == False and deep_intent.output in ["DOCUMENT_REQUEST_WITH_TOPIC"] and not state.get("innovation")) or (cl.user_session.get("resumed_conversation") == True):
                print("Intent involves innovation, allowing progression.")
                return deep_intent.output

            print("Unclear decision, requesting revision...")
            return False  # Request another review

        else:  #  LOW CONFIDENCE - Model is unsure, revision is needed
            print(f"Low agreement ({score}%), rejecting deeper intent. Reverting to intent_node...")
            return False  # Force re-evaluation


    def conversational_node(self, state: AgentState):
        """
        This function is responsible for running the conversational node. It takes in a state object 
        and runs the conversational node based on the instruction in the state. The instruction can be 
        to ask the user for refinement of its input as it was not clear for you, or to inform the user 
        that you are an agent specialized in guidance for document creation. Be concise.

        The function returns a new state object with the updated messages and the task_list and task_def 
        set to None.

        Parameters
        ----------
        state : AgentState
            The state object containing the instruction and the user's input.

        Returns
        -------
        AgentState
            The updated state object
        """
        print( "Conversational node...")

        intent  = state.get("intent", None)
        deep_intent = state.get("deep_intent", None)
        user_input  = state.get("task", None)
        message_history  = state.get("messages", None)
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        source= state.get("sources", [])
        final_doc = state.get("final_doc", {})

        manage_tasks(task_list, task_def,task_running="conversational_node", status = cl.TaskStatus.RUNNING)
        
        prompt = copy.deepcopy(CONVERSATIONAL_PROMPT).replace("$USER_INPUT$", user_input)
      
        try:
            prompt = prompt + "\n"+INTENT_TRACKING[deep_intent.output]
        except Exception as e:
            print(f"Sometime failed while preparing the output for the user. {e}")
            prompt = prompt + "\nAsk the user for refinement of its input as it was not clear for you."

        prompt = prompt + f"\nOutput only the response without your any extra comments or explanation and using the language {cl.user_session.get('language')}"
        message_history.append({"role":"user", "content" : prompt})

        response, _ = self.agent.chat(message_history, format=None, tools = None)

        manage_tasks(task_list, task_def,task_running="conversational_node", status = cl.TaskStatus.DONE)

        return {'messages': [{
                    'role': 'assistant',
                    'content': response.message.content
                }],
            "task_list" : task_list,
            "task_def" : task_def  ,
            "sources" : source}
    

    def retrieve_chunks(self,  query : str):
        
        """
        This function takes a query string and uses the retriever to find the most 
        relevant chunks of text from the knowledge base. It filters out chunks with a
        score lower than 0.3 and returns the selected chunks as a string, a list of 
        sources_elements to be presented to the user, and the list of sources.

        Parameters:
        query (str): the query string

        Returns:
        str: the selected chunks as a string
        list: the list of sources_elements to be presented to the user
        list: the list of sources
        """
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
        """
        Node responsible for generating innovative works based on user input.

        This node uses the innovation agent to generate innovative works based on user input and the conversation history.
        The node will also consider the user's critique and evaluation to refine its output.

        The node will also retrieve relevant data using the `retrieve_data_tool` if the tool is called with a valid topic.

        The node will append the output of the agent to the conversation history and update the task list with the new task status.

        :param state: The state of the conversation.
        :return: The updated state of the conversation.
        """
        print("\nInnovationg creator node ...")

        user_input  = state.get("task", None)
        conversation_history  = state.get("innovation_history", [])
        innovation_critic = state.get("innovation_critic", None)
        innovation_revision_number = state.get("innovation_revision_number", None)
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        message_history  = state.get("messages", None)



        manage_tasks(task_list, task_def,task_running="innovation_creator_node", status = cl.TaskStatus.RUNNING, revision_counter=  innovation_revision_number)
  
        tools_list = [
            self.tools_definition.retrieve_data_olama_tool(), 
            ]
        if len(conversation_history) == 0 :
            conversation_history.append({"role":"system", "content" : INNOVATION_SYSTEM_PROMPT})
            message_history.append({"role":"user", "content" : INNOVATION_SYSTEM_PROMPT})
        inputs = f"""
        - User input : {user_input}
        - **Consider the user's critique and evaluation** (`{innovation_critic}`) to refine your output.
        - **Use the language:** `{cl.user_session.get('language')}` for the final output.

        """

        conversation_history.append({"role":"user", "content" : inputs})
        message_history.append({"role":"user", "content" : inputs})

        for i in range(30):#Sometimes the ollama model output is broken , that's why we have this loop
   
            response, _ = self.tool_agent.chat(conversation_history, format=None, tools = tools_list)

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
                

            
            response, _ = self.tool_agent.chat(conversation_history, format=None, tools = tools_list)

        conversation_history.append({'role': 'assistant', 'content':  f" Output :  {response.message.content}"})
        message_history.append({"role":"assistant", "content" : f" Output :  {response.message.content}.\n"})
        manage_tasks(task_list, task_def, task_running="innovation_creator_node", status = cl.TaskStatus.DONE)

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
            "search_result" : str_sources,
            "messages" : message_history
        }

    def get_innovation_works(self, topics, sources, innovation_critic, str_sources):
        """
        Node responsible for generating innovative works based on user input and conversation history.

        The node generates a prompt for the innovation agent to generate innovative works based on user input and the conversation history.
        The node will also consider the user's critique and evaluation to refine its output.

        The node will append the output of the agent to the conversation history and update the task list with the new task status.

        :param topics: The topics for the innovation.
        :param sources: The sources for the innovation.
        :param innovation_critic: The user's critique and evaluation to refine the output.
        :param str_sources: The search result.
        :return: The output of the agent.
        """
        innovation_prompt = copy.deepcopy(INNOVATION_TOOL_PROMPT)
        innovation_prompt = innovation_prompt.replace("$TOPICS$", str(topics)).replace("$SEARCH_RESULTS$", "\n".join(sources)).replace("$CRITIC_CONVERSATION_HISTORY$", str(innovation_critic))
        response, _ = self.innovation_agent.chat([{"role":"user", "content" : innovation_prompt}], format=None, tools = None)

        output = f"""Here you have the topics : {str(topics)}
        Here you have the search result : {str_sources}
        Here you have a set of innovative works (title and description) : {str(response.message.content)}
        """
        return output

    def innovation_creator_critic_node(self, state: AgentState):
        """
        Node responsible for generating innovation critics based on user input and conversation history.

        The node generates a prompt for the innovation agent to generate innovation critics based on user input and the conversation history.
        The node will also consider the user's critique and evaluation to refine its output.

        The node will append the output of the agent to the conversation history and update the task list with the new task status.

        :param innovation: The innovation to be evaluated.
        :param conversation_history: The conversation history.
        :param task_list: The task list.
        :param task_def: The task definition.
        :param search_result: The search result.
        :return: The output of the agent.
        """
        innovation = state.get("innovation", None)
        conversation_history  = state.get("innovation_history", [])
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        search_result = state.get("search_result",None)
        manage_tasks(task_list, task_def,task_running="innovation_creator_critic_node", status = cl.TaskStatus.RUNNING)
        message_history  = state.get("messages", None)

        print("\n----------------------------")
        print(f"Getting innovation critics for : {innovation}")
        print("----------------------------")

        if innovation is None :
            raise Exception("Missing innovation ..")
        
        prompt = INNOVATION_CRITIC_PROMPT.replace("$SEARCH_RESULT$",search_result)
        conversation_history.append({'role': 'user','content': prompt })
        message_history.append({'role': 'user','content': prompt })

        _, innovative_works_critics = self.agent.chat(conversation_history, format=CriticModel, tools = None)
        conversation_history.append({'role': 'assistant', 'content':  f"Critics for the innovative works and its creation process : {str(innovative_works_critics.observation)} "})
        message_history.append({'role': 'assistant', 'content': f"Critics for the innovative works and its creation process : {str(innovative_works_critics.observation)} "})

        manage_tasks(task_list, task_def, task_running="innovation_creator_critic_node", status = cl.TaskStatus.DONE)

        return {"innovation_critic" : {
            "observation" : innovative_works_critics.observation,
            "score" : innovative_works_critics.score
        }, 
            "innovation_history" :conversation_history,
            "task_list" : task_list,
            "task_def" : task_def,
            "messages" : message_history
        }
       
    def validate_innovation_critic_decision(self, state: AgentState):
        
        """
        Validates the decision made by the innovation critic based on the score and revision number.

        This function checks if the innovation critic's score is above a threshold or if the number of
        revisions has exceeded the maximum allowed. If either condition is met, it returns True,
        indicating that the innovation critic decision is valid. Otherwise, it returns False.

        Parameters:
        state (AgentState): The current state containing the innovation critic decision and revision details.

        Returns:
        bool: True if the decision is valid, False otherwise.
        """

        innovation_critic_decision = state.get("innovation_critic", {})
        print("\n----------------------------")
        print(f"Validating innovation critic decision : {innovation_critic_decision}")
        print("----------------------------")
    
        if float(innovation_critic_decision["score"]) >= 70.0 or state["innovation_revision_number"] > state["innovation_max_revisions"]:
            return True
            
        else:
            return False
    
    def doc_creation_planning_node(self, state: AgentState):
        """
        This function is responsible for generating a document creation plan based on the user's input.
        It takes in a state object and generates a document creation plan using the provided user input.
        The function returns a new state object with the updated messages, the document creation plan and the task_list and task_def set to None.

        Parameters
        ----------
        state : AgentState
            The state object containing the user's input and the conversation history.

        Returns
        -------
        AgentState
            The updated state object
        """
        print("Document creation planning node ...")

        messages  = state.get("messages", [])
        user_input  = state.get("task", None)
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        planning= state.get("planning",{})

        manage_tasks(task_list, task_def,task_running="doc_creation_planning_node", status = cl.TaskStatus.RUNNING)
        prompt = DOC_PLANNING_PROMPT.replace("$CONVERSATION_HISTORY$", str(messages)).replace("$USER_INPUT$", str(user_input)) 

        conversation_history = [{
            'role': 'user',
            'content': prompt,
            }]
        response, planning_formated_output = self.agent.chat(conversation_history, format=DocsPlanningModel, tools = None)

        print("Complete response : ",response.message.content)

        planning = []
        for i ,section in enumerate(planning_formated_output.sections):
            planning.append({
                "title" : section.title,
                "objective" : section.section_objective,
                "subsections" : section.subsections
            })
        print("Document sections ordered : ", planning_formated_output.ordered_sections)
        print("Document selected : ", planning_formated_output.doc_title)
        print("Document selected objectives : ", planning_formated_output.doc_objective)
        print("Plan sections formatted -> ", planning)
    
        print("Updating the search result for this work  ...")
        str_sources, sources_elements, sources = self.retrieve_chunks(planning_formated_output.doc_title+" : "+ planning_formated_output.doc_objective)

        manage_tasks(task_list, task_def, task_running="doc_creation_planning_node", status = cl.TaskStatus.DONE)

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
                            "doc_objective" : planning_formated_output.doc_objective,
                            "ordered_sections" : planning_formated_output.ordered_sections
            },
            "sources" : sources_elements,
            "search_result" : str_sources
        }

    def doc_supervisor_node(self, state: AgentState):
        """
        This function is responsible for supervising the document creation process.

        It takes in a state object and generates a prompt based on the current state of the document creation process.
        The function returns a new state object with the updated messages, the document creation plan and the task_list and task_def set to None.

        Parameters
        ----------
        state : AgentState
            The state object containing the user's input and the conversation history.

        Returns
        -------
        AgentState
            The updated state object
        """
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

        manage_tasks(task_list, task_def,task_running="doc_supervisor_node", status = cl.TaskStatus.RUNNING)

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

        print(f"Order sections : {doc_details['ordered_sections']}")
        for details in doc_details["planning"]:
            if details["title"] in sections_review_control:
                try:
                    print(f" {details['title']} -> {sections_review_control[details['title']]}/{sections_max_review}")
                    if sections_review_control[details["title"]] < sections_max_review:
                        available_sections.append(details["title"])
 
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

        response, router = self.agent.chat(doc_supervisor_history, format=Router, tools = None)

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
            for details in doc_details['ordered_sections']:
                if details["title"] in doc_supervisor_outputs:
                    final_doc = final_doc + doc_supervisor_outputs[details["title"]]['results']+"\n\n"

        manage_tasks(task_list, task_def, task_running="doc_supervisor_node", status = cl.TaskStatus.DONE)

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
        """
        Evaluate if the document creation process has finished. If so, reset the state by setting back the default values for the corresponding variables.
        """
        doc_supervisor_outputs = state.get("doc_supervisor_outputs", {})

        if "next_section" in doc_supervisor_outputs:
            if doc_supervisor_outputs["next_section"] == "FINISH":
                print("Document creation finished")

                self.graph.update_state(self.thread, {
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
        """
        This node is responsible for writing a section of the document based on the objectives and observations provided by the doc supervisor.
        The node will be called multiple times until the document is finished.
        The node will be given the section_id of the section to be written and the objectives and observations of the section.
        The node should use the objectives and observations to generate the best section as possible.
        The node should return the written section and the updated conversation history.
        """
        print("Doc writer node ...")
        doc_supervisor_outputs = state.get("doc_supervisor_outputs", {})
        docs_written_sections = state.get("docs_written_sections", [])
        task_list= state.get("task_list", None)
        task_def= state.get("task_def",None)
        search_result = state.get("search_result",None)
        sections_review_control = state.get("sections_review_control", {})

        manage_tasks(task_list, task_def,task_running="doc_writer_node", status = cl.TaskStatus.RUNNING)

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
        response, _ = self.agent.chat(doc_supervisor_outputs[doc_writer_history], format=None, tools = None)
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

        manage_tasks(task_list, task_def, task_running="doc_writer_node", status = cl.TaskStatus.DONE)
        return {
            "doc_supervisor_outputs" : doc_supervisor_outputs,
            "docs_written_sections" : docs_written_sections,
            "task_list" : task_list,
            "task_def" : task_def,
            "sections_review_control" : sections_review_control
        }


#=== Receives a List of chainlit elements and appends it
def sources_update(elements):
    """
    Appends a list of elements to the list of sources stored in the user's session.
    
    Parameters
    ----------
    elements : list
        A list of elements to be appended to the list of sources.
    """
    answer_sources = cl.user_session.get("answer_sources")
    answer_sources.append(elements)
    cl.user_session.set("answer_sources", answer_sources)

#=== Receives a Chainlit Message and appends it
def update_conversation_history(msg):

    """
    Appends a Chainlit Message to the conversation history and resume history stored in the user's session.
    
    Parameters
    ----------
    msg : Chainlit Message
        The message to be appended to the conversation history.
    """
    conversation_history = cl.user_session.get("conversation_history")
    conversation_history.append(msg)
    cl.user_session.set("conversation_history", conversation_history)

    resume_history = cl.user_session.get("resume_history")
    resume_history.append({msg.author : msg.content})
    cl.user_session.set("resume_history", resume_history)


