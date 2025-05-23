#
# Copyright (c) 2025, Filip Szymanski
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

# Pipecat imports
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
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
from functions import (
    appointment_script,
    tools
)
from rag import initialize_rag_query_engine, retrieve_business_info

load_dotenv(override=True)

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

    # Store task reference for idle handler to access
    task_ref = [None]
    
    # Define a simple handler for idle user detection
    async def handle_user_idle(user_idle_processor: UserIdleProcessor, retry_count: int) -> bool:
        if retry_count == 1:
            # First attempt: Ask if they're still there
            updated_messages = list(messages)
            updated_messages.append(
                {
                    "role": "system",
                    "content": "The user has been quiet. Politely and briefly ask if they're still there.",
                }
            )
            await user_idle_processor.push_frame(LLMMessagesFrame(updated_messages))
            return True
        elif retry_count == 2:
            # Second attempt: More direct prompt
            updated_messages = list(messages)
            updated_messages.append(
                {
                    "role": "system",
                    "content": "The user is still inactive. Ask if they'd like to continue our conversation.",
                }
            )
            await user_idle_processor.push_frame(LLMMessagesFrame(updated_messages))
            return True
        else:
            # Final attempt: End the call
            logger.info("User idle timeout reached. Terminating call.")
            # Send goodbye message
            await user_idle_processor.push_frame(
                TTSSpeakFrame("It seems like you're busy right now. I'll disconnect the call. Have a nice day!")
            )
            # Wait for TTS to complete before ending
            await asyncio.sleep(5.0)
            
            # End the call by sending EndFrame directly to the task
            if task_ref[0]:
                logger.info("Sending EndFrame to terminate call")
                await task_ref[0].queue_frame(EndFrame())
                # Also cancel the task to ensure termination
                try:
                    logger.info("Cancelling task to ensure termination")
                    await task_ref[0].cancel()
                except Exception as e:
                    logger.error(f"Error cancelling task during idle termination: {e}")
            else:
                logger.error("Cannot terminate call: task reference not available")
                
            return False

    # Create the idle user processor with 5 second timeout
    user_idle = UserIdleProcessor(callback=handle_user_idle, timeout=5.0)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_idle,
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
    
    # Set the task reference for the idle handler to use
    task_ref[0] = task

    if isinstance(args, WebSocketSessionArguments):
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected: {client}")
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected: {client}")
            try:
                await task.cancel()
            except Exception as e:
                logger.error(f"Error during task cancellation: {str(e)}")

        @transport.event_handler("on_client_closed")
        async def on_client_closed(transport, client):
            logger.info(f"Client closed connection")
            try:
                await task.cancel()
            except Exception as e:
                logger.error(f"Error during task cancellation: {str(e)}")
    elif isinstance(args, DailySessionArguments):
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
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
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info("Pipeline task was cancelled. This is expected during normal call termination.")
    except Exception as e:
        logger.error(f"Pipeline task failed with an exception: {e}", exc_info=True)
        raise


async def bot(args: SessionArguments):
    try:
        await main(args)
        logger.info("Bot process completed successfully.")
    except asyncio.CancelledError:
        logger.info("Bot process was cancelled. This is expected during normal call termination.")
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
    # To enable more detailed asyncio debug logs for issues like 'tasks cancelled error':
    # asyncio.run(local(), debug=True)
    # For normal operation:
    asyncio.run(local())
