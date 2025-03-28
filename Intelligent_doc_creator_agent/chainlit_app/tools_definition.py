from typing import List, Dict, Any
import chainlit as cl

class ToolsDefinition():
   
    ################################## Translation Tool ##################################
    def translate_output_olama_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "translate_output_tool",
                "description": f"""Translate a text to the user's language.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_to_translate": {
                            "type": "string",
                            "description": "The text to be translated "
                        }
                    },
                    "required": ["text_to_translate"]
                }
            }
        }

    ################################## Innovation creation Tool ##################################
    def innovation_creator_ollama_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "innovation_creator_tool",
                "description": """Find innovative works (titles and description) to be worked on.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topics": {
                            "type": "string",
                            "description": "THe topics to be used to find innovative works"
                        }
                      
                    },
                    "required": ["topics"]
                }
            }
        }


    ################################## Document creation Tool ##################################
    def propose_research_works_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "propose_research_works_tool",
                "description": """Generative innovative works given a topic.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topics": {
                            "type": "string",
                            "description": "The topics to be used to find innovative works"
                        },
                        "search_result": {
                            "type": "string",
                            "description": "The search result from the knowldge base to be used as context for the innovative works creation"
                        }
                    },
                    "required": ["topics","search_result"]
                }
            }
        }


    # ################################## Data Retrieval Tool ##################################

    def ask_topics_olama_tool(self) -> Dict[str, Any]:
        """
        A tool that asks for missing data to the user.
        This tools is used when the user wants to create a document but did not specified the topics.
        """
        return {
            "type": "function",
            "function": {
                "name": "ask_for_topics_tool",
                "description": f"""When the user is interested in creating documents but did not specified the topics, this tool will  ask the user for the missing topics of the document to be created, using the language '{cl.user_session.get("language")}'.""",
            }
        }



    def retrieve_data_olama_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "retrieve_data_tool",
                "description": "Query and retrieve relevant information from the knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search in the knowledge base."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def document_management_tool_olama_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "document_management_tool",
                "description": """Call this tool to create the document or manage the document creation process.""",
              
            }
        }


