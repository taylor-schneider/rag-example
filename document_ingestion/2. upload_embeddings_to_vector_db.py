import os
import logging
import requests
import json
import pandas
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import hashlib
from dotenv import load_dotenv


# Configure logging
log_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(module)s:%(funcName)s():%(lineno)d] %(message)s'
logging.basicConfig(
    format=log_format,
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)


# Determine the current directory and the pdf
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)

# Load the environment variables
load_dotenv(dotenv_path=root_dir)
qdrant_api_url = os.environ["QDRANT_API_URL"]
huggingface_api_token = os.environ["HUGGING_FACE_API_TOKEN"]
openai_api_key = os.environ["OPENAI_API_KEY"]
redis_api_host = os.environ["REDIS_API_HOST"]
redis_api_port = os.environ["REDIS_API_PORT"]

# Load the embeddings
embeddings_file = "embeddings.csv"
embeddings_file_path = os.path.join(current_dir, embeddings_file)
embeddings = pandas.read_csv(embeddings_file_path, index_col=0)

# Make a connection to the vector database
logging.debug("Connecting to vector db")
client = QdrantClient(url=qdrant_api_url)

# Create a collection in the vectordb
# Specify the length of the vectors and the distance algorithm
collection_name = "test_collection"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.DOT)
    )

# Create a unique ID for the file
pdf_name = "Taylor Schneider - One Pager Accenture CV 2024 - External.pdf"
pdf_id = hashlib.sha256(pdf_name.encode()).hexdigest()

# Upsert the embeddings
points = []
for i in range(0, embeddings.shape[0]):
    vector = embeddings.iloc[i]
    point = PointStruct(id=i, vector=vector, payload={"chunk_id": i, "pdf_id": pdf_id})
    points.append(point)
    
    
operation_info = client.upsert(
    collection_name="test_collection",
    wait=True,
    points=points,
)


print(operation_info)