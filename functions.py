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

from rag import initialize_rag_query_engine, retrieve_business_info, RAG_RETRIEVER

async def appointment_script(params: FunctionCallParams):
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