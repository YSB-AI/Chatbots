from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import  AnyMessage, SystemMessage, HumanMessage, ToolMessage
import operator
import chainlit as cl
from langgraph.graph import StateGraph, START, END


intent_options = ["NORMAL", "DOCUMENTS"]

class UserIntentModel(BaseModel):
    output: Literal[*intent_options]
    rationale : str

class CriticModel(BaseModel):
    observation : str
    recommendations : str
    score : int

class DocSectionsModel(BaseModel):
    title : str
    section_objective : str
    subsections: dict  

class DocsPlanningModel(BaseModel):
    doc_title : str
    doc_objective : str
    sections: List[DocSectionsModel]
    ordered_sections: list


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


def manage_tasks(task_list, task_def, task_running, status, revision_counter = None):
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
    
    if task_list is not None and len(task_list.tasks) > 0:
        
        task_id = list(task_json)[0]
        task_list.tasks[task_id].status = status

        if revision_counter is not None and revision_counter >1: 
            task_list.tasks[task_id].title = task_json[task_id] + f" (Revision {revision_counter})"