
import os
import pandas
import requests
import time
import qdrant_client
from jinja2 import Environment, BaseLoader
from openai import OpenAI
import redis
from dotenv import load_dotenv
from qdrant_client import QdrantClient


class Agent:
  
  def __init__(self):

    # Load the environment variables
    load_dotenv()
    self.qdrant_api_url = os.environ["QDRANT_API_URL"]
    self.huggingface_api_token = os.environ["HUGGING_FACE_API_TOKEN"]
    self.openai_api_key = os.environ["OPENAI_API_KEY"]
    self.redis_api_host = os.environ["REDIS_API_HOST"]
    self.redis_api_port = os.environ["REDIS_API_PORT"]

    # Set some other configurations
    self.vector_db_collection_name = "test_collection"
    self.vector_db_client = QdrantClient(url=self.qdrant_api_url)
    
    # Connect to datastores
    self.openai_client = OpenAI(
      api_key=self.openai_api_key,
    )
    self.redis_client = redis.Redis(host='15.4.50.102', port=6379)

  
  def generate_embedding(self, text):
    
      model_id = "sentence-transformers/all-MiniLM-L6-v2"
      api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
      headers = {"Authorization": f"Bearer {self.huggingface_api_token}"}
      
      # Sometimes HF fails and we will need to retry the api call
      # We will hard code three attempts
      for i in range(0,3):
        try:
          response = requests.post(api_url, 
                                  headers=headers, 
                                  json={"inputs": [text], "options":{"wait_for_model":True}},
                                  verify=False)
          return response.json()[0]
        except Exception as e:
          if i == 2:
            raise e
          time.sleep(2)
  
  def get_chunks_related_to_query(self, query_vector, max_chunks):

    # Search the vector db for the three closest matches
    search_results = self.vector_db_client.search(
        collection_name=self.vector_db_collection_name, 
        query_vector=query_vector, 
        limit=max_chunks
    )

    best_match_chuck_ids = [(search_result.payload["chunk_id"], search_result.payload["pdf_id"]) for search_result in search_results]
    
    chunks = []
    for best_match_chuck_id in best_match_chuck_ids:
        chunk_id = best_match_chuck_id[0]
        pdf_id = best_match_chuck_id[1]
        chunk = self.redis_client.json().get(f"docs:{pdf_id}")["chunks"][chunk_id]
        chunks.append(chunk)
    
    return chunks

  def generate_prompt(self, relevant_chunks, user_query):
    prompt_template = """
You are performing retrieval augmented generation.
When answering questions you must use the relevant information contained
in the following three chunks:

Chunk 1:
{{ relevant_chunks[0] }}

Chunk 2:
{{ relevant_chunks[1] }}

Chunk 3:
{{ relevant_chunks[2] }}

The question to answer is:
{{ user_query }}
"""
    jinja_template = Environment(loader=BaseLoader).from_string(prompt_template)
    jinja_variables = {
      "relevant_chunks": relevant_chunks,
      "user_query": user_query
    }
    prompt = jinja_template.render(**jinja_variables)
    return prompt

  def prompt_llm(self, prompt):
    
    completion = self.openai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": prompt},
      ]
    )

    response = completion.choices[0].message.content
    return response

# ===========================================
# Main Loop
# ===========================================

agent = Agent()
while True:
  print("What would you like to know about the document?")
  user_query = input()
  #user_query = "What are Taylor Schneider's top 5 qualities?"

  query_vector = agent.generate_embedding(user_query)
  relevant_chunks = agent.get_chunks_related_to_query(query_vector=query_vector, max_chunks=3)
  prompt = agent.generate_prompt(relevant_chunks=relevant_chunks, user_query=user_query)
  response = agent.prompt_llm(prompt)
  print(response)

 
