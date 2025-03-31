import os

import chainlit as cl
from chainlit.input_widget import Select, Slider
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import ollama
from chainlit.types import ThreadDict
import time
from typing import Dict, Optional
from dotenv import load_dotenv
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from fpdf import FPDF
import fpdf
import unicodedata
 #https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from langchain.schema.runnable import RunnableConfig

load_dotenv()

import sys
sys.path.append('../../')

hf_token = os.getenv("HF_TOKEN", None) 
model_id = os.getenv("model_id", None) 
embed_model_name = os.getenv("embed_model_name", None)
llama_model_id = os.getenv("llama_model_id", None)

host = os.getenv("host", None)
user = os.getenv("user", None)
password = os.getenv("password", None)
port = os.getenv("port", None)
database= os.getenv("database", None)
table_name= os.getenv("table_name", None)

BUCKET_NAME = os.getenv("BUCKET_NAME","STORAGE") 
TOOLS_METHOD  =  os.getenv("TOOLS_METHOD", None)
conninfo = f"postgresql+asyncpg://{user}:{password}@{host}:5432/chainlit"

tasks_definition = {
"intent_node": "Getting intent" ,
"intent_deeper_node" : "Getting deeper intent" ,
"intent_critic_node" : "Criticizing deeper intent" ,
"innovation_creator_node" : "Getting Innovation" ,
"innovation_creator_critic_node" : "Criticizing Innovation" ,
"conversational_node" : "Preparing interaction" ,
"doc_creation_planning_node" : "Document planning" ,
"doc_supervisor_node" : "Document supervisor" ,
"doc_writer_node" : "Document writer"
}

from FSStorageClient import *
from local_bucket import *

os.environ["TOKENIZERS_PARALLELISM"]="true"
############### Create a virtual storage client #########################

# Configure data layer
#https://github.com/Chainlit/chainlit/issues/1205
fs_storage_client = FSStorageClient(
    storage_path=os.path.join(os.getcwd(), BUCKET_NAME),
    url_path=BUCKET_NAME
)
cl.data._data_layer = SQLAlchemyDataLayer(conninfo=conninfo, storage_provider=fs_storage_client)

from llama_index.core.bridge.pydantic import BaseModel, Field
from typing import Optional
import chainlit as cl

from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent

embedding_model = HuggingFaceEmbedding(model_name=embed_model_name)

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.checkpoint.memory import MemorySaver

os.environ["LANGGRAPH_DEFAULT_RECURSION_LIMIT"]="50"

from ai_agent import *
#from ai_custom_multiagent import *


DOCUMENT_ROOT_PATH = "./DOCUMENTS/"
os.makedirs(DOCUMENT_ROOT_PATH, exist_ok=True)
print(f"Directory '{DOCUMENT_ROOT_PATH}' created or already exists.")



@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin") : # Just for testing locally. Ideally this should not be hardcoded
        return cl.User(
            identifier=username, metadata={"role": username, "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def start():
    print("Starting conversation...")

      # Get the user session
    # Foundation model
    llm = HuggingFaceInferenceAPI(
        model_name=model_id, 
        tokenizer_name=model_id, 
        token=hf_token,
        context_window = 100000,
        num_outputs = Field(
                default=4000,
                description="The number of tokens to generate.",
            ),
        is_chat_model = True,
        generate_kwargs={
            "do_sample": True,
            "temperature": 0,
            "top_p": 0.9,
        },
    )

    cl.user_session.set("model", ollama)

    # Embedding model
    cl.user_session.set("embed_model", embedding_model)
    
    # Configs
    
    # # bge embedding model
    Settings.embed_model = embedding_model

    # # Llama-3-8B-Instruct model
    Settings.llm = llm

    pgvector_store = PGVectorStore.from_params(
        database = database,
        host = host,
        password = password,
        port = port,
        user = user,
        table_name = table_name, 
        embed_dim = 1024,  # selected model embedding dimension
        hybrid_search=True, # Hybrid vector search

    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=pgvector_store, 
        use_async = True
        )

    semantic_retriever = index.as_retriever(similarity_top_k=50)
    
    sparse_retriever= index.as_retriever(
        vector_store_query_mode="sparse", sparse_top_k=10
        )

    fusion_retriever = QueryFusionRetriever(
            [semantic_retriever],#, sparse_retriever],
            similarity_top_k=20,
            num_queries=2,  # set this to 1 to disable query generation
            mode="relative_score",
            use_async=True,
        )
    response_synthesizer = CompactAndRefine()
    query_engine = RetrieverQueryEngine(
        retriever=fusion_retriever,
        response_synthesizer=response_synthesizer,
    )
    retriever = semantic_retriever
    # Use the index as a retriever
    
    cl.user_session.set("retriever", retriever)
    stream_thread = {"configurable": {"thread_id": "1"}}
    
    cl.user_session.set("agent_threads", stream_thread)
    cl.user_session.set("answer_sources", [])
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("resume_history", [])
    cl.user_session.set("user_input", "")
    cl.user_session.set("temperature_param", 0.9)
    cl.user_session.set("language", "PT-PT")
    cl.user_session.set("most_recent_history",[])
    cl.user_session.set("message_history", [])
    cl.user_session.set("current_running_task", None)
    cl.user_session.set("resumed_conversation", False)

    # Attach the saver to your graph
    #memory = AsyncSqliteSaver.from_conn_string(":memory:")
    memory = MemorySaver()
    agent = AIAgent(
        retriever=retriever,
        checkpointer=memory,
        stream_thread = stream_thread
    )
    cl.user_session.set("agent", agent)
        
    # except Exception as e:
    #     print("Erro init variables: ", e)

    # Settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="Language",
                label="Language",
                values=["PT", "EN"],
                initial_index=0,
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0.9,
                min=0,
                max=1,
                step=0.05,
            ),
        ]
    ).send()
    
    text_msg = "Olá, sou o teu Assistente Virtual pessoal. Em que posso ajudar ?"
    msg = cl.Message(
        author="Assistant", content=text_msg
    )
    await msg.send()

    agent.graph.update_state(stream_thread, {"messages" : [{"role":"assistant", "content" : text_msg}]})

    cl.user_session.set("agent", agent)

    update_conversation_history(msg)
    sources_update(msg.elements)


@cl.on_settings_update
async def setup_agent(settings):

    print("on_settings_update: ", settings)

    if settings["Language"] == "PT":
        cl.user_session.set("language", "PT-PT")
    elif settings["Language"] == "EN":
        cl.user_session.set("language", "EN-GB")

    if float(settings["Temperature"]) !=  float(cl.user_session.get("temperature_param")):
        print(f"Temperature parameter changed from {float(cl.user_session.get('temperature_param'))} to {float(settings['Temperature'])}")
        # Update temperature
        cl.user_session.set("temperature_param", settings["Temperature"])
 

@cl.action_callback("Sources")
async def on_action(action):

    print("Action: ", action)
    count = 0
    while count < 10:
        chat = cl.user_session.get("conversation_history")
        try:
            print("chat length | action +1  | Count",len(chat), int(action.value) +1, count)
            if len(chat)> int(action.value):
                break
        except Exception as e:
            print(f"Failed to get chat history on action. {e}")
            
        time.sleep(1)
        count +=1
    
    print("on_action()  chat history and length: ",chat, len(chat))
    
    msg = chat[int(action.value)]
    
    print("on_action()  chat message content : ", msg.content)
    
    if len( msg.elements ) == 0:
        print("answer_sources and length : ",cl.user_session.get("answer_sources"), len(cl.user_session.get("answer_sources")))
        msg.elements = cl.user_session.get("answer_sources")[int(action.value)]
    else:
        msg.elements = []
    chat[int(action.value)] = msg
    
    cl.user_session.set("conversation_history", chat)
    
    await msg.update()
    
    await action.remove()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    llm = HuggingFaceInferenceAPI(
        model_name=model_id, 
        tokenizer_name=model_id, 
        token=hf_token,
        context_window = 100000,
        num_outputs = Field(
                default=4000,
                description="The number of tokens to generate.",
            ),
        is_chat_model = True,
        generate_kwargs={
            "do_sample": True,
            "temperature": 0,
            "top_p": 0.9,
        },
    )

    cl.user_session.set("model", ollama)

    # Embedding model
    cl.user_session.set("embed_model", embedding_model)
    
    # Configs
    
    # # bge embedding model
    Settings.embed_model = embedding_model

    # # Llama-3-8B-Instruct model
    Settings.llm = llm

    pgvector_store = PGVectorStore.from_params(
        database = database,
        host = host,
        password = password,
        port = port,
        user = user,
        table_name = table_name, 
        embed_dim = 1024,  # selected model embedding dimension
        hybrid_search=True, # Hybrid vector search

    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=pgvector_store, 
        use_async = True
        )

    semantic_retriever = index.as_retriever(similarity_top_k=50)
    
    sparse_retriever= index.as_retriever(
        vector_store_query_mode="sparse", sparse_top_k=10
        )

    fusion_retriever = QueryFusionRetriever(
            [semantic_retriever],#, sparse_retriever],
            similarity_top_k=20,
            num_queries=2,  # set this to 1 to disable query generation
            mode="relative_score",
            use_async=True,
        )
    response_synthesizer = CompactAndRefine()
    query_engine = RetrieverQueryEngine(
        retriever=fusion_retriever,
        response_synthesizer=response_synthesizer,
    )
    retriever = semantic_retriever
    # Use the index as a retriever
    
    cl.user_session.set("retriever", retriever)
    stream_thread = {"configurable": {"thread_id": "1"}}
    
    cl.user_session.set("agent_threads", stream_thread)
    cl.user_session.set("answer_sources", [])
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("resume_history", [])
    cl.user_session.set("user_input", "")
    cl.user_session.set("temperature_param", thread["metadata"]["temperature_param"])
    cl.user_session.set("language", thread["metadata"]["language"])
    cl.user_session.set("most_recent_history",thread["metadata"]["most_recent_history"])
    cl.user_session.set("message_history", thread["metadata"]["message_history"])
    cl.user_session.set("current_running_task", None)
    cl.user_session.set("resumed_conversation", True)
    

 
    memory = MemorySaver()
    agent = AIAgent(
        retriever=retriever,
        checkpointer=memory,
        stream_thread = stream_thread
    )
    cl.user_session.set("agent", agent)
    
    
    for i in range(len(thread["metadata"]["resume_history"])):
        role = list(thread["metadata"]["resume_history"][i])[0]
        print(role,":",thread["metadata"]["resume_history"][i][role])

        msg = cl.Message(content = thread["metadata"]["resume_history"][i][role], author=role)
    
        update_conversation_history(msg)
        sources_update([])
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Language",
                label="Language",
                values=["PT", "EN"],
                initial_index=0,
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=thread["metadata"]["temperature_param"],
                min=0,
                max=1,
                step=0.05,
            ),
        ]
    ).send()



@cl.on_message
async def main(message: cl.Message):

    # Creating tasks

    task_list = cl.TaskList()
    task_list.status = "Running..."
    task_retries_control = {}
    task_def = {}
    task_process_list = []
    for i, (task_process, description) in enumerate(tasks_definition.items()):
        task = cl.Task(title=description) 
        await task_list.add_task(task)
        
        task_process_list.append(task_process)
        task_def[task_process] = {
            i : description
        }

    # Getting most recent conversation history only
    conversation_history = cl.user_session.get("conversation_history")
    history = []
    n_historic_data = 8
    for i,hist_msg in enumerate(conversation_history):
        if i%2 ==0:
            history.append({"role":"assistant", "content" : hist_msg.content})
        else:
            history.append({"role":"user", "content" : hist_msg.content})
    
    history = history[-n_historic_data:]
    cl.user_session.set("most_recent_history", history)

    # Udate the complete conversation history with the user's message
    update_conversation_history(message)
    sources_update(message.elements) 

    print("New Message ---------> ", message.content)
    cl.user_session.set("user_input", message.content)

    agent = cl.user_session.get("agent")
    # Stream the output message

    thread = cl.user_session.get("agent_threads")
    msg = cl.Message(content="",  author="Assistant", actions = []) 

    sources = []
    task_running = None
    innovation_stream = False
    doc_stream = False


    try:   
     
        async for event in agent.graph.astream_events({
                    "task": message.content,
                    "messages" : [{"role":"user", "content" : message.content}],
                    "intent_revision_number" : 1,
                    "intent_max_revisions" : 5,
                    "innovation_revision_number" : 1,
                    "innovation_max_revisions" : 3,
                    "task_list" : task_list,
                    "task_def" : task_def,
                    "sections_max_review" : 5,
                    "sections_review_control": {}
                    }, 
                    thread,
                    version="v1"):

            
            if "event" in event and "name" in event:

                if event["event"] in ["on_chain_start","on_chain_end"] and event["name"] in task_process_list:
                    print(f"Updating tasks for {event['event']} event in {event['name']} node ...")
                    task_running = event['name']
                    task_list = event["data"]["input"]["task_list"]
                    task_def = event["data"]["input"]["task_def"]
                    await task_list.send()

            if "event" in event :
                

                doc_node = "doc_supervisor_node"#"doc_writer_node"
                if event["event"] == "on_chain_stream":

                    if "data" in event:
                        if "chunk" in event["data"]:
                            if "conversational_node" in event["data"]["chunk"]:
                                if "messages" in event["data"]["chunk"]["conversational_node"]:
                                    messages = event["data"]["chunk"]["conversational_node"]["messages"]
                                    sources = event["data"]["chunk"]["conversational_node"]["sources"]
                                    if len(messages)> 0 :
                                        if "content" in messages[-1]:
                                            final_msg = messages[-1]["content"]
                                                
                                            print("Final answer : ", final_msg)
                                            if final_msg:
                                                if innovation_stream :
                                                    msg.content = ""
                                                    await msg.update()
                                                    innovation_stream = False

                                                for token in final_msg:
                                                    try:
                                                        await msg.stream_token(token)
                                                    except Exception as e:
                                                        print(f"Failed to retrieve the token {token}. {e}")
                                                
                                                break
                            elif doc_node in event["data"]["chunk"]:
                                if "doc_supervisor_outputs" in event["data"]["chunk"][doc_node]:
                                    docs_written_sections = event["data"]["chunk"][doc_node]["docs_written_sections"]
                                    final_doc = event["data"]["chunk"][doc_node]["final_doc"]
                                    doc_details= event["data"]["chunk"][doc_node]["doc_details"]
                                    doc_title = doc_details["doc_title"]

                                    print("docs_written_sections ->",docs_written_sections)
                                    print("final_doc ->",final_doc)
                                    if len(docs_written_sections)> 0 and final_doc != "" :
                                        if doc_stream :
                                            msg.content = ""
                                            await msg.update()

                                        doc_path = f"{DOCUMENT_ROOT_PATH}/{doc_title.strip().replace(' ','_')}.pdf"
                                        print(f" Creating file : {doc_path}")
                                        

                                        normalized_title = unicodedata.normalize("NFKC", doc_title)
                                        normalized_content = unicodedata.normalize("NFKC", final_doc)
                                            # Create PDF with Unicode support
                                        pdf = FPDF()
                                        pdf.set_auto_page_break(auto=True, margin=15)
                                        
                                        pdf.add_page()
                                        
                                        # Use DejaVu font for Unicode support
                                        pdf.set_font("Arial", "B", 16)  # Bold font for title
                                        pdf.multi_cell(0, 10, normalized_title)  # Add title
                                        pdf.ln(10)  # Add space after title
                                        

                                        pdf.set_font("Arial", size=12)  # Regular font for content
                                        pdf.multi_cell(0, 10, normalized_content, 0, 'L', False)
                                        pdf.output(doc_path, "F")

                                        for token in final_doc:
                                            try:
                                                await msg.stream_token(token)
                                            except Exception as e:
                                                print(f"Failed to retrieve the token {token}. {e}")
                                        
                        

    except Exception as e:
        print("Failed to run the stream : ",e)
        print("Task running : ",task_running)
        if task_running is not None:
            task_json = task_def[task_running]
            task_id = list(task_json)[0]
            task_list.tasks[task_id].status = cl.TaskStatus.FAILED

        task_list.status = "Failed"
        await cl.sleep(0.2)
        await task_list.send()

    

    sources_update(sources)
    context = cl.user_session.get("answer_sources")[int(len(cl.user_session.get("conversation_history")))]
    if len(context)> 0 :
        msg.actions = [cl.Action(name="Sources", value= str(len(cl.user_session.get("conversation_history"))))]
        await msg.update()
    
    # Update the conversation history
    cl.user_session.set("agent", agent)
    update_conversation_history(msg)

    agent.graph.update_state(thread, {
        "innovation_revision_number": 1,
        "intent_revision_number": 1,
        "innovation_history" : [],
        #"intent_history" : [],
        "sources" : [],
        "search_result" : "",
        "doc_planning" : "",
        "doc_supervisor_outputs" : {},
        "doc_supervisor_history" : []
        })
    




@cl.on_chat_end
def end():
    print("Finalizing the conversation...")
