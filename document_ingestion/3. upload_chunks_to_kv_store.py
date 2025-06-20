import os
import logging
import requests
import json
import hashlib
from dotenv import load_dotenv
import redis


# Configure logging
log_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(module)s:%(funcName)s():%(lineno)d] %(message)s'
logging.basicConfig(
    format=log_format,
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)


# Determine the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))

# Load the environment variables
load_dotenv(dotenv_path=root_dir)
qdrant_api_url = os.environ["QDRANT_API_URL"]
huggingface_api_token = os.environ["HUGGING_FACE_API_TOKEN"]
openai_api_key = os.environ["OPENAI_API_KEY"]
redis_api_host = os.environ["REDIS_API_HOST"]
redis_api_port = os.environ["REDIS_API_PORT"]


# Load the chunks
chunk_file = "chunks.json"
chunk_file_path = os.path.join(current_dir, chunk_file)
with open(chunk_file_path, "r") as fp:
    chunks = json.load(fp)

# Create a unique ID for the file
pdf_name = "Taylor Schneider - One Pager Accenture CV 2024 - External.pdf"
pdf_id = hashlib.sha256(pdf_name.encode()).hexdigest()

# Make a connection to the keyvalue store
logging.debug("Connecting to kv store")
from redis.commands.search.field import TextField
r = redis.Redis(host=redis_api_host, port=redis_api_port)

doc = {
    "name": pdf_name,
    "chunks": chunks
}

r.json().set(f"docs:{pdf_id}", "$", doc)

# Grab a specific chunk
# r.json().get(f"docs:{pdf_id}")["chunks"][3]
