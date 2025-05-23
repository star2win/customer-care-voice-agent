FROM dailyco/pipecat-base:latest

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./bot.py bot.py
COPY ./functions.py functions.py
COPY ./prompts.md prompts.md
COPY ./rag.py rag.py
COPY ./runner.py runner.py
COPY ./server.py server.py
COPY ./templates templates
COPY ./knowledge_base_docs knowledge_base_docs