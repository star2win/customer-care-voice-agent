# Customer Service Voice Agent for Automotive Business

## Overview
A telephony based customer service agent built with Pipecat and LlamaIndex, utilizing a STT->LLM->TTS pipeline.  Make a call, learn about the business, and schedule an appointment.

Features include:
* A prompt file "prompts.md" that contains the system prompt and initial greeting.
* Questions about the BME business are answered via LlamaIndex RAG.  The knowledge base is stored in the "knowledge_base_docs" directory.  You may drop additional files in this directory to expand the knowledge base.
* Appointment scheduling is handled by calling a webhook hosted on n8n with information about the appointment request.  The agent currently does not check for availability on the calendar.  The n8n webhook is configured to send an email to the BME inbox.
* The agent listens for a pause, and nudges the caller twice to speak.  If the caller does not speak, the agent will end the call with a polite message.
* The deployment options include:
    * Running the bot locally and connecting via Twilio by starting the script with "python bot.py local".
    * Running the bot locally, with a room in Daily Cloud opened dynamically, by starting the script with "python bot.py daily".
    * Dockerizing the application and running it on Daily Cloud.

## 📋 Prerequisites

* Python 3.13.3
* Pipecat
* LlamaIndex
* Deepgram
* OpenAI
* ElevenLabs (or Cartesia, need to uncomment)

## Optional
* Docker
* Twilio
* Daily / Pipecat Cloud (for Daily Room or Docker deployment)

## 🛠️ Getting Started

Copy the env.example file to .env and fill in the values with your API keys and credentials.

Set up a virtual environment before following these instructions. From the root of the repo:

   ```shell
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the development dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. Run the bot locally with Twilio:

   ```shell
   python bot.py local
   ```

   Set up ngrok to forward the webhook to the server running the bot.

   Configure Twilio with ngrok URL as the webhook.

   Edit the streams.xml file to point to the ngrok URL.

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
       <Connect>
         <Stream url="wss://<your ngrok url>>/ws" />
      </Connect>
   </Response>
   ```

4. Run the bot locally with a room in Daily Cloud:

   ```shell
   python bot.py daily
   ```

   Copy the URL displayed after starting bot.py and paste into your browser to start conversation.

5. Host the bot with Docker on Daily  (Twilio optional):

   ```shell
   docker build --platform=linux/arm64 -t customer-service-voice-agent:latest .
   docker tag customer-service-voice-agent:latest your-username/customer-service-voice-agent:latest
   docker push your-username/customer-service-voice-agent:latest
   ```

   Deploy the bot to Daily:

   ```shell
   pcc auth login # to authenticate
   pcc secrets set customer-service-voice-agent-secrets --file .env # to store your environment variables
   pcc deploy customer-service-voice-agent your-username/customer-service-voice-agent:latest --secrets customer-service-voice-agent-secrets
   ```

   Create a TwiML Bin on Twilio with the following values from Daily Cloud:
   
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
    <Connect>
        <Stream url="wss://api.pipecat.daily.co/ws/twilio">
        <Parameter name="_pipecatCloudServiceHost" value="customer-service-voice-agent.ORGANIZATION_NAME"/>
        </Stream>
    </Connect>
   </Response>