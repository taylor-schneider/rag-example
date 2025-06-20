import os
import logging
import requests
import json
from dotenv import load_dotenv

# Configure logging
log_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(module)s:%(funcName)s():%(lineno)d] %(message)s'
logging.basicConfig(
    format=log_format,
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

# Load the environment variables
load_dotenv()
pdf_inspector_url = os.environ["PDF_INSPECTOR_URL"]
pdf_file_name = os.environ["PDF_FILE_NAME"]

# Determine the current directory and the pdf
current_dir = os.path.abspath(os.path.dirname(__file__))
pdf_name = "Taylor Schneider - One Pager Accenture CV 2024 - External.pdf"
pdf_path = os.path.join(current_dir, pdf_name)

# Check that the pdf exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(pdf_path)

# Upload the pdf to the pdf inspector and get the text
files = {
    'file': open(pdf_path, 'rb'),
}
upload_url = f'{pdf_inspector_url}/get_text'
upload_response = requests.put(upload_url, files=files)
upload_result = json.loads(upload_response.content.decode())
file_name = upload_result["file_name"]
extraction_result = upload_result["extraction_result"]

# Write the text to a file
output_file = "extract.txt"
output_path = os.path.join(current_dir, output_file)
with open(output_path, "w") as fp:
    #json.dump(extraction_result, fp)
    fp.write(extraction_result["text"])