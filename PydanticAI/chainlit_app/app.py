import os

import chainlit as cl
from chainlit.input_widget import Select, Slider
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import ollama
import time
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
 #https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.bridge.pydantic import  Field
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from chainlit.types import ThreadDict

from doc_sources_management import *

load_dotenv()

import sys
sys.path.append('../../')

hf_token = os.getenv("HF_TOKEN", None) 
model_id = os.getenv("model_id", None) 
embed_model_name = os.getenv("embed_model_name", None)

host = os.getenv("host", None)
user = os.getenv("user", None)
password = os.getenv("password", None)
port = os.getenv("port", None)
database= os.getenv("database", None)
table_name= os.getenv("table_name", None)

BUCKET_NAME = os.getenv("BUCKET_NAME","STORAGE") 
TOOLS_METHOD  =  os.getenv("TOOLS_METHOD", None)
conninfo = f"postgresql+asyncpg://{user}:{password}@{host}:5432/chainlit"

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

embedding_model = HuggingFaceEmbedding(model_name=embed_model_name)

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


os.environ["LANGGRAPH_DEFAULT_RECURSION_LIMIT"]="50"

from ai_agent import *

tasks_definition = {
"Thinking" : "Thinking and making decisions" ,
"critique_interactor_agent" :"Criticizing the decision taken",
"retrieve_data" : "Retrieving data",
"doc_planning" : "Creating a plan for document creation" ,
"doc_creation_process" : "Supervising document creation process" ,
"doc_writer_process" : "Document writing process" ,
"doc_update_process" : "Updating a section of the document" ,
"process_output" : "Processing the output"
}

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

    semantic_retriever = index.as_retriever(similarity_top_k=100)
    st_rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-12-v2",  # A good balance of speed and quality
        top_n=20  # Return top 5 reranked results
    )

    # Create a response synthesizer
    response_synthesizer = CompactAndRefine()

    # Create the query engine
    query_engine = RetrieverQueryEngine(
        retriever=semantic_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[st_rerank]
    )
    
    # Use the index as a retriever
    
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
    cl.user_session.set("doc_planning_json", None)
    cl.user_session.set("agent_history", None)
    

    # Attach the saver to your graph
    agent = AIAgent(
        retriever=query_engine#, llm = llm
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

    cl.user_session.set("agent", agent)

    update_conversation_history(msg)
    sources_update(msg.elements)

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
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

    semantic_retriever = index.as_retriever(similarity_top_k=100)
    st_rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-12-v2",  # A good balance of speed and quality
        top_n=20  # Return top 5 reranked results
    )

    # Create a response synthesizer
    response_synthesizer = CompactAndRefine()

    # Create the query engine
    query_engine = RetrieverQueryEngine(
        retriever=semantic_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[st_rerank]
    )
    
    # Use the index as a retriever
    
    cl.user_session.set("answer_sources", [])
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("resume_history", [])
    cl.user_session.set("user_input", "")
    cl.user_session.set("temperature_param", thread["metadata"]["temperature_param"])
    cl.user_session.set("language", thread["metadata"]["language"])
    cl.user_session.set("most_recent_history",thread["metadata"]["most_recent_history"])
    cl.user_session.set("message_history", thread["metadata"]["message_history"])
    cl.user_session.set("current_running_task", None)
    cl.user_session.set("resumed_conversation", False)
    cl.user_session.set("doc_planning_json", thread["metadata"]["doc_planning_json"])
    cl.user_session.set("agent_history", None)
 
        # Attach the saver to your graph
    agent = AIAgent(
        retriever=query_engine#, llm = llm
    )
    agent.doc_planning_json = thread["metadata"]["doc_planning_json"]
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
        #print("answer_sources and length : ",cl.user_session.get("answer_sources"), len(cl.user_session.get("answer_sources")))
        msg.elements = cl.user_session.get("answer_sources")[int(action.value)]
    else:
        msg.elements = []
    chat[int(action.value)] = msg
    
    cl.user_session.set("conversation_history", chat)
    
    await msg.update()
    
    await action.remove()
    


@cl.on_message
async def main(message: cl.Message):

    # Creating tasks

    task_list = cl.TaskList()
    task_list.status = "Running..."
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
    n_historic_data = 10
    for i,hist_msg in enumerate(conversation_history):
        if i%2 ==0:
            history.append({hist_msg.content})
        else:
            history.append({"user : " : hist_msg.content})
    
    history = history[-n_historic_data:]
    cl.user_session.set("most_recent_history", history)

    # Udate the complete conversation history with the user's message
    update_conversation_history(message)
    sources_update(message.elements) 

    print("New Message ---------> ", message.content)
    cl.user_session.set("user_input", message.content)

    agent = cl.user_session.get("agent")
    agent.task_list = task_list
    agent.task_def = task_def

    final_msg = await agent.generate_response(message.content, history)

    await agent.manage_tasks(agent.task_list, agent.task_def,task_running="process_output", status = cl.TaskStatus.RUNNING )
    sources_update(agent.sources_doc)

    if len(agent.sources_doc)> 0 :
        msg  = cl.Message(content="",  author="Assistant", actions = [cl.Action(name="Sources", value= str(len(cl.user_session.get("conversation_history"))))])
    else:
        msg = cl.Message(content="",  author="Assistant") 

    for token in final_msg:
        try:
            await msg.stream_token(token)
        except Exception as e:
            print(f"Failed to retrieve the token {token}. {e}")

    await agent.manage_tasks(agent.task_list, agent.task_def,task_running="process_output", status = cl.TaskStatus.DONE )
    await msg.send()

    # Update the conversation history
    agent.doc_creation_revision_counter = 1
    agent.doc_writer_revision_counter = 1
    agent.final_output_revision_counter = 1
    agent.task_list = None
    agent.task_def = None
    agent.sources_doc = []
    
    cl.user_session.set("agent", agent)

    update_conversation_history(msg)


@cl.on_chat_end
def end():
    print("Finalizing the conversation...")
