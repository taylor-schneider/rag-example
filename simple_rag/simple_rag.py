# ===========================================
# Initialize
# ===========================================
import os
import pandas
import json
import redis

# ===========================================
# Initialize
# ===========================================

# Figure out the paths to our files
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)
doc_ingestion_dir = os.path.join(root_dir, "document_ingestion")

# Load the environment variables
from dotenv import load_dotenv
load_dotenv()
qdrant_api_url = os.environ["QDRANT_API_URL"]
huggingface_api_token = os.environ["HUGGING_FACE_API_TOKEN"]
openai_api_key = os.environ["OPENAI_API_KEY"]
redis_api_host = os.environ["REDIS_API_HOST"]
redis_api_port = os.environ["REDIS_API_PORT"]

# ===========================================
# Connect to data stores
# ===========================================

# Connect to vectordb
from qdrant_client import QdrantClient
client = QdrantClient(url=qdrant_api_url)

# Make a conection to the chunk kv store
r = redis.Redis(host=redis_api_host, port=redis_api_port)

# ===========================================
# Run Agent Loop
# ===========================================

while True:

    # ===========================================
    # Embed The Query
    # ===========================================

    # Create a query and find the chunk that matches best
    print("What would you like to know about the document?")
    #user_query = "What are Taylor Schneider's top 5 qualities?"
    user_query = input()

    # Embed the query
    import requests
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {huggingface_api_token}"}

    def query(texts):
        response = requests.post(api_url,
                                headers=headers,
                                json={"inputs": texts, "options":{"wait_for_model":True}},
                                verify=False)
        return response.json()

    output = query([user_query])

    query_vector = output[0]

    # ===========================================
    # Find chunks related to query
    # ===========================================

    # Search the vector db for the three closest matches
    search_results = client.search(
        collection_name="test_collection", query_vector=query_vector, limit=3
    )

    #print(search_results)

    # Get the corresponding chunk and document IDs
    best_match_chuck_ids = [(search_result.payload["chunk_id"], search_result.payload["pdf_id"]) for search_result in search_results]
    #print(best_match_chuck_ids)

    # ===========================================
    # Generate Prompt and send to llm
    # ===========================================


    
    # Get the chunks coresponding to the best matching embeddings
    chunks = []
    for best_match_chuck_id in best_match_chuck_ids:
        chunk_id = best_match_chuck_id[0]
        pdf_id = best_match_chuck_id[1]
        chunk = r.json().get(f"docs:{pdf_id}")["chunks"][chunk_id]
        chunks.append(chunk)

    # Embed the chunks into the prompt
    system_prompt = f"""
    You are performing retrieval augmented generation.
    When answering questions you must use the relevant information contained
    in the following three chunks:

    Chunk 1:
    {chunks[0]}

    Chunk 2:
    {chunks[1]}

    Chunk 3:
    {chunks[2]}

    The question to answer is:
    """

    # Send the prompt to open ai
    from openai import OpenAI
    openai_client = OpenAI(
        api_key=openai_api_key,
    )

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    # ===========================================
    # Show the llm response
    # ===========================================

    print(completion.choices[0].message.content)