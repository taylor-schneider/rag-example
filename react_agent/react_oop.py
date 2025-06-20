
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
from sentence_transformers import SentenceTransformer
import re
import json
import traceback

import warnings

# Ignore all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Determine the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)


class Agent:
  
  def __init__(self):

    # Load the environment variables
    load_dotenv(root_dir)
    self.qdrant_api_url = os.environ["QDRANT_API_URL"]
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
    self.redis_client = redis.Redis(host=self.redis_api_host, port=self.redis_api_port)
    
    # Create an instance of the embedding model
    full_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_name = 'all-MiniLM-L6-v2'
    model_path = os.path.join(root_dir, ".models", model_name)
    if not os.path.exists(model_path):
      self.embedding_model = SentenceTransformer(full_model_name)
      self.embedding_model.save(model_path)
    
    self.embedding_model = SentenceTransformer(model_path)
    
    # Load the prompt template
    prompt_path = os.path.join(current_dir, "prompt.j2")
    with open(prompt_path, 'r') as file:
      self.prompt_template = file.read()


  # ========================================
  # Tool Functions
  # ========================================

  def get_tool_descriptions(self):
    return """
get_contact_info() - get Taylor's full name, title, and email address.
get_background_info() - get a brief summary of Taylor's background.
list_skillsets() - get a list of Taylor's skillsets.
list_tools() - get a list of tools that Taylor has experience with.
list_employers_before_accenture() - get a list of places where Taylor worked prior to joining accenture.
search_resume(query) - lookup detailed work experience information from taylor's resume.
"""

  def get_contact_info(self):
    return "name: Taylor Schneider, level: Senior Manager, email: taylor.schneider@accenture.com"

  def get_background_info(self):
    return """Taylor is a seasoned Solutions Architect based out of
Chicago. He has spent 10+ years building automation
and facilitating organizational change. Taylor specializes
in ML and AI Ops. He co-founded the global
MLE/MLOps COP within Accenture. Taylor has a wide
skillset having served as a developer, engineer,
sysadmin, architect, and manager. Taylor’s soft skills
allow him to effectively communicate and collaborate
with both front and back of house. 
"""

  def list_skillsets(self):
    return ["Software Architecture", "DevOps/MLOps/GenAI", "Container Orchestration", "Distributed Systems", "CI/CD/CT", "Scrum Master", "Windows/Linux"]
  
  def list_tools(self):
    return ["Docker/Kubernetes", "AWS/Azure/Databricks", "Ansible/Terraform", "PowerShell/Bash", "C#/Python", "Jenkins/Azure DevOps"]
  
  def list_employers_before_accenture(self):
    return ["Federal Home Loan Bank", "Bank Of America Merrill Lynch", "FactSet Research Systems"]
  
  def search_resume(self, query):
    query_vector = agent.generate_embedding(query)
    relevant_chunks = agent.get_chunks_related_to_query(query_vector=query_vector, max_chunks=3)
    return relevant_chunks
  

  # ========================================
  # Utility Functions
  # ========================================
  
  def generate_embedding(self, text):
      embedding = self.embedding_model.encode(text)
      return embedding
  
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

  def generate_prompt(self):

    tool_descriptions = self.get_tool_descriptions()
    jinja_template = Environment(loader=BaseLoader).from_string(self.prompt_template)
    jinja_variables = {
      "tool_descriptions": tool_descriptions
    }
    prompt = jinja_template.render(**jinja_variables)
    return prompt

  def prompt_llm(self, prompt, conversation):
    
    completion = self.openai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system", "content": prompt
        },
        {
          "role": "user", "content": self.conversation_to_str(conversation)
        }
      ]
    )

    response = completion.choices[0].message.content
    return response

# ===========================================
# ReAct Loop
# ===========================================

  def turn_to_str(self, turn):
    turn_str = ""
    for key, value in turn.items():
      turn_str += f"{key}: {value}" + "\n"
    return turn_str.strip("\n")

  def conversation_to_str(self, conversation):
    return "\n".join([self.turn_to_str(turn) for turn in conversation])

  def react(self, user_query, loop_limit=5):
    conversation = []

#    conversation.append(f"QUESTION: {user_query}")
#    print(conversation[-1])

    conversation.append({"QUESTION": user_query})


    for i in range(0, loop_limit):
      
      # Print the prior turn
      print(self.turn_to_str(conversation[i]))
      
      print("-----------------------")
      
      
      turn = {}
      
      # Prompt the llm based on the latest conversation
      prompt = self.generate_prompt()
      llm_response = self.prompt_llm(prompt, conversation).strip()
#      print(llm_response)
      
      # Get a list of tokens returned from the llm
      tokens = re.findall("\n?[A-Z]+:", llm_response)
      tokens = [token.strip("\n").strip(":") for token in tokens]
      
      # Make sure they match what we are expecting
      if tokens != ["REASON", "ANSWER"] and tokens != ["REASON", "ACTION"] and tokens != ["REASON", "ACTION", "PARAMS"]:
        turn["ERROR"] = f"The llm did not ReAct correctly. It returned the following token set: {json.dumps(tokens)}."
        conversation.append(turn)
        continue
      
      # Check that we got a reason
      reason = None
      try:
        reason = re.search("REASON:.*(ACTION|ANSWER)", llm_response, re.DOTALL).group()
        reason = re.sub("^REASON: ", "", reason).strip()
        reason = re.sub("(ACTION|ANSWER)$", "", reason).strip()
        turn["REASON"] = reason
      except:
        turn["ERROR"] = f"The llm did not ReAct correctly. It did not return a REASON."
        conversation.append(turn)
        continue
      
      # Check if we got an answer
      answer = None
      try:
        answer = re.search("ANSWER:.*", llm_response, re.DOTALL).group()
        answer = re.sub("^ANSWER: ", "", answer).strip()
        turn["ANSWER"] = answer
        print(self.turn_to_str(turn))
        conversation.append(turn)
        return conversation
      except:
        pass
      
      # Otherwise if we got a reason/act, lets try to execute
      
      # Parse out the tool selection and parameter
      try:
        action = re.search("ACTION:.*(\n)?", llm_response).group()
        action = re.sub("^ACTION: ", "", action).strip()
        action = re.sub("\\((.*)?\\)", "", action)
        if not action or action == "None":
          raise
        turn["ACTION"] = action
      except Exception as ex:
        turn["OBSERVE"] = "No action was specified. Please try to REASON again."
        conversation.append(turn)
        continue
      
      try:
        param_match = re.search("PARAMS:.*(\n)?", llm_response)
        param_line = param_match.group() if param_match else ""
        param_line = re.sub("^PARAMS: ", "", param_line).strip()
        params = json.loads(param_line) if param_line else {}
        turn["PARAMS"] = params
      except Exception as e:
        turn["OBSERVE"] = "Unable to parse the supplied params. Please ensure they are proper json."
        conversation.append(turn)
        continue
      
      # Try to get a pointer to the tool function
      try:
        method = getattr(self, turn["ACTION"])
      except Exception as e:
        turn["OBSERVE"] = f"The tool selection failed because the tool name was not correct. The tool '{action}' does not exist. Please try a different ACTION."
        conversation.append(turn)
        continue
      
      # Invoke the tool function
      tool_result = ""
      try:
        if params:
          tool_result = method(**params)
        else:
          tool_result = method()
      except Exception as e:
        tool_result = traceback.format_exc()
        turn["OBSERVE"] = f"Executing the ACTION failed. Please check the PARAMS are correct. The error message was as follows:" + "\n" + tool_result + "\n" + "Try the ACTION again with different parameters."
        conversation.append(turn)
        continue
      
      turn["OBSERVE"] = tool_result
      conversation.append(turn)

    print(self.turn_to_str(conversation[-1]))
    
    print("-----------------------")

    turn = {}
    turn["ERROR"] = f"Unable to answer question in {loop_limit} turns."
    print(self.turn_to_str(turn))

    conversation.append(turn)
    return conversation
          
# ===========================================
# Main Loop
# ===========================================

agent = Agent()
conversation = []
while True:
  print("===============================")
  print("What is your question about Taylor?")
  user_query = input()
  print("-----------------------")
  #user_query = "What are the main skillsets of AIOps?"

  agent.react(user_query)

