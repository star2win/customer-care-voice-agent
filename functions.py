import asyncio
import json
import os
from pathlib import Path

import aiohttp
from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from rag import initialize_rag_query_engine, retrieve_business_info

# --- LlamaIndex RAG Setup START ---
KNOWLEDGE_BASE_DIR = "knowledge_base_docs"  # Directory for your knowledge documents
RAG_RETRIEVER = None  # Global variable for the retriever

def initialize_rag_query_engine():
    global RAG_RETRIEVER

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager  # Apply it globally to LlamaIndex Settings

    if RAG_RETRIEVER is None:
        logger.info("Initializing LlamaIndex RAG retriever...")
        try:
            # Configure LlamaIndex to use OpenAI models for LLM and Embeddings
            Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo")
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
            Settings.num_output = 256
            Settings.context_window = 3900

            # Check if knowledge base directory exists
            kb_path = Path(KNOWLEDGE_BASE_DIR)
            if not kb_path.exists() or not any(kb_path.iterdir()):
                logger.warning(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' is empty or does not exist.")
                logger.warning("RAG will not be able to answer questions from the knowledge base.")
                documents = []
            else:
                reader = SimpleDirectoryReader(KNOWLEDGE_BASE_DIR)
                documents = reader.load_data()

            if not documents:
                logger.warning("No documents found in the knowledge base. RAG might not be effective.")
                RAG_RETRIEVER = "empty"  # Special placeholder
                return

            index = VectorStoreIndex.from_documents(documents)
            RAG_RETRIEVER = index.as_retriever(similarity_top_k=2)  # Retrieve top 2 similar nodes for better context
            logger.info("LlamaIndex RAG retriever initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex RAG retriever: {e}")
            RAG_RETRIEVER = None

async def retrieve_business_info(params: FunctionCallParams):
    """
    Retrieves relevant information from the business knowledge base using LlamaIndex
    based on a user's query.
    """
    if RAG_RETRIEVER is None:
        logger.error("RAG Retriever is not initialized.")
        await params.result_callback({
            "status": "error",
            "message": "Knowledge base search is currently unavailable (retriever not initialized).",
            "retrieved_snippets": []
        })
        return
    
    if RAG_RETRIEVER == "empty":
        logger.warning("RAG Retriever is empty (no documents loaded).")
        await params.result_callback({
            "status": "info_not_found",
            "message": "No information found in the knowledge base.",
            "retrieved_snippets": []
        })
        return

    query = params.arguments.get("query", "")
    if not query:
        await params.result_callback({
            "status": "error",
            "message": "Query cannot be empty for knowledge base search.",
            "retrieved_snippets": []
        })
        return

    logger.info(f"RAG: Received query: {query}")
    try:
        # Use aiohttp.loop.run_in_executor for synchronous LlamaIndex call
        loop = asyncio.get_event_loop()
        # LlamaIndex retrieve is synchronous, so run it in an executor
        retrieved_nodes = await loop.run_in_executor(None, RAG_RETRIEVER.retrieve, query)
        
        if retrieved_nodes:
            snippets = [node.get_content() for node in retrieved_nodes]
            logger.info(f"RAG: Retrieved {len(snippets)} snippets")
            logger.debug(f"RAG: First snippet preview: {snippets[0][:200]}...")

            await params.result_callback({
                "status": "success",
                "message": "Information retrieved successfully.",
                "retrieved_snippets": snippets
            })
        else:
            await params.result_callback({
                "status": "info_not_found",
                "message": "No specific information found for your query in the knowledge base.",
                "retrieved_snippets": []
            })
    except Exception as e:
        logger.error(f"Error retrieving from LlamaIndex: {e}")
        await params.result_callback({
            "status": "error",
            "message": f"An error occurred while searching the knowledge base: {str(e)}",
            "retrieved_snippets": []
        })

async def appointment_script(params: FunctionCallParams):
    # await params.llm.push_frame(TTSSpeakFrame("I'm waiting to confirm the appointment request."))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://nextoutcome.app.n8n.cloud/webhook/ca0b9164-e451-4f67-8d16-3d2659a99c26",
                json={
                    "name": params.arguments.get("name"),
                    "phone": int(params.arguments.get("phone")),
                    "make": params.arguments.get("make"),
                    "model": params.arguments.get("model"),
                    "year": int(params.arguments.get("year")),
                    "day": params.arguments.get("day"),
                    "problem": params.arguments.get("problem"),
                    "summary": params.arguments.get("summary"),
                },
            ) as response:
                if response.status == 200:
                    response_text = await response.text()
                    if "appointment" in response_text.lower():
                        await params.result_callback({"status": "success", "message": "The appointment request has been sent successfully."})
                    else:
                        await params.result_callback({"status": "error", "message": f"Unexpected response from appointment service: {response_text}"})
                else:
                    await params.result_callback({"status": "error", "message": f"Failed to send appointment request: {response.status}"})
    except Exception as e:
        await params.result_callback({"status": "error", "message": f"Failed to send appointment request: {str(e)}"})

# Function schemas
retrieve_business_info_schema = FunctionSchema(
    name="retrieve_business_info",
    description="Looks up information about Bavarian Motor Experts (services, hours, location, policies, specific details etc.) from the company's knowledge base. Use this tool when the user asks a question that isn't about scheduling an appointment and requires specific business details. The tool returns raw text snippets from the knowledge base that you should use to formulate your response.",
    properties={
        "query": {
            "type": "string",
            "description": "The user's question or the specific topic they are asking about the business.",
        }
    },
    required=["query"],
)

appointment_script_schema = FunctionSchema(
    name="appointment_script",
    description="Tool makes an appointment by sending a request to n8n software",
    properties={
        "name": {
            "type": "string",
            "description": "First and Last name of the customer",
        },
        "phone": {
            "type": "integer",
            "description": "Phone number of the customer",
        },
        "make": {
            "type": "string",
            "description": "Car make",
        },
        "model": {
            "type": "string",
            "description": "Car model",
        },
        "year": {
            "type": "integer",
            "description": "Car year",
        },
        "day": {
            "type": "string",
            "description": "Preferred day to service car",
        },
        "problem": {
            "type": "string",
            "description": "Car problem description",
        },
        "summary": {
            "type": "string",
            "description": "Call summary including conversation history",
        },
    },
    required=["name", "phone", "make", "model", "year", "day", "problem", "summary"],
)

# Tools schema
tools = ToolsSchema(standard_tools=[appointment_script_schema, retrieve_business_info_schema]) 