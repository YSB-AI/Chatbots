# Virtual Assistant & Intelligent Document Writer

## Overview

This repository contains three AI-powered projects designed to enhance conversational AI and document generation using advanced retrieval and multi-agent techniques.

## Projects

### 1. Virtual Assistant with RAG

A virtual assistant capable of handling both general conversations and domain-specific queries related to scientific research. Additionally, it can generate structured academic-style documents using a multi-agent system.

**Features:**

- General conversational capabilities

- Domain-specific scientific query handling

- Structured document generation following academic paper conventions



### 2. Intelligent Document Writer

An AI-driven assistant that guides users through creating large documents. Users provide topics, and the system retrieves relevant information, suggests innovative ideas, plans the document structure, generates content for each section, critiques the results against the plan, and exports the final document as a PDF.

**Features:**

- User-guided document creation

- Information retrieval and idea generation

- Multi-agent workflow for structured content generation

- Auto-critique and refinement of document sections

- PDF export of the final document

## Technologies Used

- Ollama models with tool support

- LangGraph – Graph-based framework for managing multi-agent workflows

- RAG (Retrieval-Augmented Generation) – Enhancing responses with relevant scientific information

- Python 3

- Chainlit – UI for interactive AI applications

- PostgreSQL + pgvector – Efficient vector storage for retrieval




### 3. Intelligent Assistant and Document Writer

An AI-driven assistant capable of handling both general conversations and domain-specific queries related to scientific research, as well as guiding the user through creating and updating large documents. In this implementation the user is capable of not only create documents but also performing changes to the suggested document plan and the final document. This is essentially a mix between the first two projects but adding the updates capability.

**Features:**

- General conversational capabilities

- Domain-specific scientific query handling

- Structured document generation following academic paper conventions

- User-guided document creation

- Information retrieval and idea generation

- Plan creation and updating

- Multi-agent workflow for structured content generation

- Auto-critique and refinement of document sections

- PDF export of the final document

- Updates to the created document

## Technologies Used

- Google Gemini 2.0  with tool support

- Pydantic AI

- RAG (Retrieval-Augmented Generation) – Enhancing responses with relevant scientific information

- Python 3

- Chainlit – UI for interactive AI applications

- PostgreSQL + pgvector – Efficient vector storage for retrieval