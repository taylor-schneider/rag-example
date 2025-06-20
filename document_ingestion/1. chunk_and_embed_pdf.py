import os
import logging
import requests
import json
import pandas as pd
from sentence_transformers import SentenceTransformer


# Configure logging
log_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(module)s:%(funcName)s():%(lineno)d] %(message)s'
logging.basicConfig(
    format=log_format,
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)


# Determine the current directory and the extract path
current_dir = os.path.abspath(os.path.dirname(__file__))
extract_file = "extract.txt"
extract_path = os.path.join(current_dir, extract_file)
root_dir = os.path.dirname(current_dir)

# Read the text from the file
with open(extract_path, "r") as fp:
    text = fp.read()


# Divide the document text into chunks of equal word count
words = text.split(" ")
word_count = len(words)
chunk_count = 10
chunks = []
chunk_size = round(word_count / chunk_count)
for i in range(0, chunk_count - 1):
    chunk = " ".join(words[i * chunk_size:(i+1)*chunk_size])
    chunks.append(chunk)
chunks.append(" ".join(words[chunk_count - 1 * chunk_size:]))


# Write the chunks to a file
chunk_file = "chunks.json"
chunk_file_path = os.path.join(current_dir, chunk_file)
with open(chunk_file_path, "w") as fp:
    json.dump(chunks, fp, indent=4)

# Create an local instance of the embedding model
logging.debug("Creatig local embedding model")
model_name = 'all-MiniLM-L6-v2'
model_path = os.path.join(root_dir, ".models", model_name)
model = SentenceTransformer(model_path)

# Convert the text into embeddings using the huggingface model
embeddings = model.encode(chunks)

# Load embeddings into a dataframe
embeddings_df = pd.DataFrame(embeddings)

# Write the embeddings df to file
embeddings_file = "embeddings.csv"
embeddings_path = os.path.join(current_dir, embeddings_file)
embeddings_df.to_csv(embeddings_path)

