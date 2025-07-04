import base64
import os
from langchain_community.llms import Cohere
from langchain_openai import AzureChatOpenAI
import pandas as pd 
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import tqdm
import json

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
def describe_image(image_path):
    base64_image = encode_image(image_path)
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """Using the poster, explain the movie style, characteristics, emotions the person can feel while watching the movie.
    Do not make up an answer, present just pure fact what the picture shows. Describe it in up to 100 words.
    """
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ],
    }
    DIAL_API_KEY = os.environ["DIAL_API_KEY"]
    llm = AzureChatOpenAI(
        api_version="2024-10-21",
        azure_endpoint="https://ai-proxy.lab.epam.com",
        api_key=DIAL_API_KEY,
        model="gpt-4o")

    response = llm.invoke([message])
    return response.text()

def load_poster_description(folder_path):
    documents = []
    for item_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, item_name)
        if os.path.isfile(image_path):  
            poster_desc = describe_image(image_path)
            filename_with_extension = os.path.basename(image_path)
            filename_without_extension = os.path.splitext(filename_with_extension)[0]
            print(filename_without_extension)
            print(poster_desc)
            metadata = {                
                "imdb_id": filename_without_extension        
            }
            documents.append(Document(page_content=poster_desc, metadata=metadata))
        
    # Initialize the embedding model
    embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Create the vector store
    persist_directory = "data/vectorstore"
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    vector_store.add_documents(documents)

import time

# Calculate the number of seconds in 3 hours
# 3 hours * 60 minutes/hour * 60 seconds/minute
wait_seconds = 3 * 60 * 60

folder_path = "data/Posters2" 
load_poster_description(folder_path)   
