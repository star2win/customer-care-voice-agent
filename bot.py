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
# --- LlamaIndex Imports END ---

load_dotenv(override=True)

# --- LlamaIndex RAG Setup START ---
KNOWLEDGE_BASE_DIR = "knowledge_base_docs"  # Directory for your knowledge documents
RAG_QUERY_ENGINE = None # Global variable for the query engine

def initialize_rag_query_engine():
    global RAG_QUERY_ENGINE
    if RAG_QUERY_ENGINE is None:
        logger.info("Initializing LlamaIndex RAG query engine...")
        try:
            # Configure LlamaIndex to use OpenAI models for LLM and Embeddings
            # It will use OPENAI_API_KEY from environment variables
            Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo") # Can be gpt-4o too, but 3.5-turbo is faster/cheaper for RAG synthesis
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small") # Or "text-embedding-ada-002"
            Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
            Settings.num_output = 256 # Max tokens for LLM response in RAG
            Settings.context_window = 3900 # For gpt-3.5-turbo

            # Check if knowledge base directory exists
            kb_path = Path(KNOWLEDGE_BASE_DIR)
            if not kb_path.exists() or not any(kb_path.iterdir()):
                 logger.warning(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' is empty or does not exist.")
                 logger.warning("RAG will not be able to answer questions from the knowledge base.")
                 # You might want to create an empty index or handle this case differently
                 # For now, we'll let it proceed, and the retriever will find nothing.
                 documents = []
            else:
                reader = SimpleDirectoryReader(KNOWLEDGE_BASE_DIR)
                documents = reader.load_data()

            if not documents:
                logger.warning("No documents found in the knowledge base. RAG might not be effective.")
                # Create an empty index if no documents are found to prevent errors later
                # This is a simplistic way to handle it; a more robust solution might be needed.
                RAG_QUERY_ENGINE = "empty" # Special placeholder
                return

            index = VectorStoreIndex.from_documents(documents)
            RAG_QUERY_ENGINE = index.as_query_engine(similarity_top_k=3) # Retrieve top 3 similar nodes
            logger.info("LlamaIndex RAG query engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex RAG query engine: {e}")
            RAG_QUERY_ENGINE = None # Ensure it's None if initialization fails

async def retrieve_business_info(params: FunctionCallParams):
    """
    Retrieves relevant information from the business knowledge base using LlamaIndex
    based on a user's query.
    """
    if RAG_QUERY_ENGINE is None:
        logger.error("RAG Query Engine is not initialized.")
        await params.result_callback({
            "status": "error",
            "message": "Knowledge base search is currently unavailable (engine not initialized).",
            "retrieved_snippets": []
        })
        return
    
    if RAG_QUERY_ENGINE == "empty":
        logger.warning("RAG Query Engine is empty (no documents loaded).")
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
        # LlamaIndex query is synchronous, so run it in an executor
        response = await loop.run_in_executor(None, RAG_QUERY_ENGINE.query, query)
        
        retrieved_text = str(response) # response.response is also common
        source_nodes_texts = [node.get_content() for node in response.source_nodes]

        logger.info(f"RAG: Retrieved response: {retrieved_text}")
        logger.debug(f"RAG: Source nodes count: {len(response.source_nodes)}")
        # for i, node in enumerate(response.source_nodes):
        #     logger.debug(f"RAG Source Node {i+1} (Score: {node.score:.4f}):\n{node.get_content()[:200]}...")


        if retrieved_text and "empty query result" not in retrieved_text.lower() and "i don't know" not in retrieved_text.lower():
             # LlamaIndex query engine's response IS the synthesized answer.
             # We will pass this directly as the "snippet" for the main LLM to use.
             # Or, you could have the main LLM just relay this if it's good enough.
             # For simplicity, we'll pass it as if it's a retrieved chunk.
            await params.result_callback({
                "status": "success",
                "message": "Information retrieved successfully.",
                "retrieved_snippets": [retrieved_text] # Pass the synthesized answer as a single snippet
            })
        else:
            await params.result_callback({
                "status": "info_not_found",
                "message": "No specific information found for your query in the knowledge base.",
                "retrieved_snippets": []
            })
    except Exception as e:
        logger.error(f"Error querying LlamaIndex: {e}")
        await params.result_callback({
            "status": "error",
            "message": f"An error occurred while searching the knowledge base: {str(e)}",
            "retrieved_snippets": []
        })

retrieve_business_info_schema = FunctionSchema(
    name="retrieve_business_info",
    description="Looks up information about Bavarian Motor Experts (services, hours, location, policies, specific details etc.) from the company's knowledge base. Use this tool when the user asks a question that isn't about scheduling an appointment and requires specific business details.",
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
    await params.llm.push_frame(TTSSpeakFrame("I'll send the appointment request now."))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://nextoutcome.app.n8n.cloud/webhook/ca0b9164-e451-4f67-8d16-3d2659a99c26",
                json={
                    "name": params.arguments.get("name"),
                    "phone": int(params.arguments.get("phone")),
                    "make": params.arguments.get("make"),
                    "model": params.arguments.get("model"),
                    "year": params.arguments.get("year"),
                    "day": params.arguments.get("day"),
                    "problem": params.arguments.get("problem"),
                    "summary": params.arguments.get("summary"),
                },
            ) as response:
                if response.status == 200:
                    await params.result_callback({"status": "success", "message": "The appointment request has been sent"})
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
            "type": "string",
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

    # Register the basketball scores functions
    llm.register_function("appointment_script", appointment_script)
    llm.register_function("retrieve_business_info", retrieve_business_info)

    messages = [
        {
            "role": "system",
            "content": """You are a friendly and efficient virtual assistant for Bavarian Motor Experts.
                You speak short sentences, are brief, and listen to the caller.
                Your role is to assist customers by answering questions about the company's services, schedule appointments, and take messages.
                You should use the provided knowledge base to offer accurate and helpful responses by using the 'retrieve_business_info' tool for any questions about the business itself (e.g. hours, services, location, policies).
                The 'retrieve_business_info' tool will return a synthesized answer based on the knowledge base. You should use this answer to respond to the user.

                <tasks>
                - Answer Questions: If the question is about Bavarian Motor Experts (e.g., services, hours, location, specific policies), use the 'retrieve_business_info' tool with the user's query to get information. The tool will provide an answer; use this answer to respond to the user. If the tool indicates no information was found, politely state that you don't have that specific detail.
                - Clarify Unclear Requests: Politely ask for more details if the customer's question is not clear, before attempting to use any tool.
                - Make appointments for car service with 'appointment_script' tool.  Follow <appointment> script.
                </tasks>

                <guidelines>
                - Maintain a friendly and professional tone throughout the conversation.
                - Be patient and attentive to the customer's needs.
                - If the 'retrieve_business_info' tool returns no information, politely state that you don't have that specific detail.
                - Avoid discussing topics unrelated to the company's products or services.
                - Aim to provide concise answers. Limit responses to a couple of sentences and let the user guide you on where to provide more detail.
                - Do not repeat what the customer said.
                - Do not repeat yourself.
                - Do not confirm more than once.
                - Pronounce each number individually.  For example 8870 would be pronounced eight, eight, seven, zero and not eighty eight seventy.
                - When taking action like to run script, inform caller you are contacting the office for a follow-up while you send message.  The goal is not to have a long pause and have the caller wonder what is happening.
                - IMPORTANT: Only call the appointment_script tool ONCE after ALL required information has been collected.
                - When using 'retrieve_business_info', formulate a natural language query for the tool based on the user's question.
                </guidelines>

                <appointment>
                Ask for the details below, one at a time, when scheduling an appointment, and inform that BME will call back next business day to confirm exact day and time for the appointment.
                IMPORTANT: Only call the appointment_script tool ONCE after ALL of these details have been collected:
                - First and Last name
                - Ask if a phone call to {{system__caller_id}} works.  If {{system__caller_id}} is 0 ask the preferred phone number
                - Car make, model, year
                - Preferred day to service car
                - Car problem description
                
                After collecting ALL information, call the appointment_script tool ONCE with all the details.
                Wait for its response, informing the caller that you have reached out and are waiting for a response.
                Once you get confirmation from appointment_script that the scheduling message was sent, let the user know it's done and hang up with 'end_call' tool.
                </appointment>""",
        },
        {
            "role": "system",
            "content": "Start the conversation with:  Hi, I'm the Bavarian Motor Experts virtual agent.  How may I help you?",
        },
    ]

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
