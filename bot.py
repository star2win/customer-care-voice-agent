#
# Copyright (c) 2025, Filip Szymanski
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import datetime
import json
import os
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecatcloud.agent import (
    DailySessionArguments,
    SessionArguments,
    WebSocketSessionArguments,
)

from runner import configure

# --- LlamaIndex Imports START ---
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding # Corrected import
from llama_index.llms.openai import OpenAI as LlamaOpenAI # Corrected import, aliased to avoid conflict
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler # Timing handler
# --- LlamaIndex Imports END ---

load_dotenv(override=True)

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
# --- LlamaIndex RAG Setup END --- 

async def appointment_script(params: FunctionCallParams):
#    await params.llm.push_frame(TTSSpeakFrame("I'm waiting to confirm the appointment request."))
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

tools = ToolsSchema(standard_tools=[appointment_script_schema, retrieve_business_info_schema])

def load_prompts():
    """Load prompts from the markdown file."""
    try:
        with open("prompts.md", "r") as f:
            content = f.read()
            
        # Split the content into sections
        sections = content.split("##")
        
        # Extract system prompt (first section after the title)
        system_prompt = sections[1].split("\n", 1)[1].strip()
        
        # Extract initial greeting (second section)
        initial_greeting = sections[2].split("\n", 1)[1].strip()
        
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "system",
                "content": initial_greeting
            }
        ]
    except Exception as e:
        logger.error(f"Failed to load prompts: {e}")
        # Fallback to default prompts if file reading fails
        return [
            {
                "role": "system",
                "content": "You are a friendly and efficient virtual assistant for Bavarian Motor Experts. How may I help you?"
            }
        ]

async def main(args: SessionArguments):
    # Initialize RAG query engine once
    initialize_rag_query_engine()

    if isinstance(args, WebSocketSessionArguments):
        logger.debug("Starting WebSocket bot")

        start_data = args.websocket.iter_text()
        await start_data.__anext__()
        call_data = json.loads(await start_data.__anext__())
        stream_sid = call_data["start"]["streamSid"]
        transport = FastAPIWebsocketTransport(
            websocket=args.websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=TwilioFrameSerializer(stream_sid),
            ),
        )
    elif isinstance(args, DailySessionArguments):
        logger.debug("Starting Daily bot")
        transport = DailyTransport(
            args.room_url,
            args.token,
            "Respond bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                transcription_enabled=False,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    # Register the functions
    llm.register_function("appointment_script", appointment_script)
    llm.register_function("retrieve_business_info", retrieve_business_info)

    # Load prompts from markdown file
    messages = load_prompts()

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    if isinstance(args, WebSocketSessionArguments):
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected: {client}")
            # Kick off the conversation
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected: {client}")
            await task.cancel()

        @transport.event_handler("on_client_closed")
        async def on_client_closed(transport, client):
            logger.info(f"Client closed connection")
            await task.cancel()
    elif isinstance(args, DailySessionArguments):
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant}")
            try:
                await task.cancel()
            except Exception as e:
                # Ignore Mediasoup consumer errors during cleanup
                if "ConsumerNoLongerExists" not in str(e):
                    logger.error(f"Error during cleanup: {str(e)}")
                    raise
                logger.debug("Ignoring Mediasoup consumer cleanup error")

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(args: SessionArguments):
    try:
        await main(args)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise


async def local():
    async with aiohttp.ClientSession() as session:
        if os.getenv("DAILY_API_KEY"):
            (room_url, token) = await configure(session)

            await main(
                DailySessionArguments(
                    session_id=None,
                    room_url=room_url,
                    token=token,
                    body=None,
                )
            )

        elif os.getenv("DAILY_ROOM_URL") and os.getenv("DAILY_TOKEN"):
            await main(
                DailySessionArguments(
                    session_id=None,
                    room_url=os.getenv("DAILY_ROOM_URL"),
                    token=os.getenv("DAILY_TOKEN"),
                    body=None,
                )
            )

        else:
            logger.error(
                "DAILY_ROOM_URL and DAILY_TOKEN must be set in your .env file to use Daily."
            )


if __name__ == "__main__":
    asyncio.run(local())
