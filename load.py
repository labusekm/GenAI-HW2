import pandas as pd 
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import tqdm
import json


def load_netflix(vector_store):
    documents = []
    data = pd.read_csv("data/netflix.csv", sep=";", encoding='cp1252', on_bad_lines='skip') 
    data = data[data["show_id"].astype(str).str.match(r"s\d+")]
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "title": x["title"],
            "type": x["type"],            
            "age": int(x["rating"].replace("+", "") if isinstance(x["rating"], str) else x["rating"]),
            "netflix":1
        }
    }, axis=1)

    # Convert to list of dicts
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
        
    vector_store.add_documents(documents)


def load_amazon(vector_store):
    documents = []
    data = pd.read_csv("data/amazon_prime_titles.csv", sep=",", on_bad_lines='skip') 
    data = data[data["show_id"].astype(str).str.match(r"s\d+")]
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "type": x["type"],            
            "age": int(x["rating"].replace("+", "") if isinstance(x["rating"], str) else x["rating"]),
            "amazon":1
        }
    }, axis=1)

    # Convert to list of dicts
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
        
    vector_store.add_documents(documents)

def load_amazon2(vector_store):
    documents = []
    data = pd.read_csv("data/amazon2.csv", sep=",", on_bad_lines='skip') 
    data = data[data["show_id"].astype(str).str.match(r"s\d+")]
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "type": x["type"],            
            "age": int(x["rating"].replace("+", "") if isinstance(x["rating"], str) else x["rating"]),
            "amazon":1
        }
    }, axis=1)

    # Convert to list of dicts
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
        
    vector_store.add_documents(documents)

def load_disney(vector_store):
    documents = []
    data = pd.read_csv("data/disney_movies.csv",  sep=",",  on_bad_lines='skip') 
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "type": 'Movie',
            "disney":1
        }
    }, axis=1)
    # Convert to list of dicts
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
        
    vector_store.add_documents(documents)  

def load_disney_plus(vector_store):
    documents = []
    data = pd.read_csv("data/disney_plus_titles.csv",  sep=",",  on_bad_lines='skip') 
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "type": x["type"],
            "disney":1
        }
    }, axis=1)
    # Convert to list of dicts
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
        
    vector_store.add_documents(documents)

def load_MoviesOnStreamingPlatforms(vector_store):
    documents = []
    data = pd.read_csv("data/MoviesOnStreamingPlatforms.csv",  sep=",", on_bad_lines='skip') 
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "type": 'Movie',
            "age": int(x["Age"].replace("+", "") if isinstance(x["Age"], str) else x["Age"]),       
            "netflix": x["Netflix"],     
            "disney": x["Disney+"],    
            "amazon": x["Prime Video"],
            "hulu": x["Hulu"],
        }
    }, axis=1)
    # Convert to list of dicts
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
        
    vector_store.add_documents(documents)

def load_MoviesOnStreamingPlatforms2(vector_store):
    documents = []
    data = pd.read_csv("data/platforms2.csv",  sep=",", on_bad_lines='skip') 
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "type": 'Movie',
            "age": int(x["Age"].replace("+", "") if isinstance(x["Age"], str) else x["Age"]),       
            "netflix": x["Netflix"],     
            "disney": x["Disney+"],    
            "amazon": x["Prime Video"],
            "hulu": x["Hulu"],
        }
    }, axis=1)
    # Convert to list of dicts
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))
        
    vector_store.add_documents(documents)

def load_Imdb(vector_store):
    documents = []
    data = pd.read_csv("data/imdb_movies.csv", sep=",", on_bad_lines='skip') 
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "title": x["names"],
            "type": 'Movie',
        }
    }, axis=1)
        
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))

    vector_store.add_documents(documents)

def load_Imdb2(vector_store):
    documents = []
    data = pd.read_csv("data/imdb2.csv", sep=",", on_bad_lines='skip') 
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "title": x["names"],
            "type": 'Movie',
        }
    }, axis=1)
        
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))

    vector_store.add_documents(documents)

def load_Imdb3(vector_store):
    documents = []
    data = pd.read_csv("data/imdb3.csv", sep=",", on_bad_lines='skip') 
    transformed = data.apply(lambda x: { 
        "page_content": json.dumps(x.to_dict()), 
        "metadata": {
            "title": x["names"],
            "type": 'Movie',
        }
    }, axis=1)
        
    result = transformed.tolist()

    # Create documents from the transformed data
    for item in result:
        documents.append(Document(page_content=item["page_content"], metadata=item["metadata"]))

    vector_store.add_documents(documents)


# Initialize the embedding model
embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Create the vector store
persist_directory = "data/vectorstore"
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# print('load_netflix')
# load_netflix(vector_store)
# print('load_amazon')
# load_amazon(vector_store)
# print('load_amazon2')
# load_amazon2(vector_store)
# print('load_disney')
# load_disney(vector_store)
# print('load_disney_plus')
# load_disney_plus(vector_store)
# print('load_MoviesOnStreamingPlatforms')
# load_MoviesOnStreamingPlatforms(vector_store)
# print('load_MoviesOnStreamingPlatforms2')
# load_MoviesOnStreamingPlatforms2(vector_store)
print('load_Imdb')
load_Imdb(vector_store)
print('load_Imdb2')
load_Imdb2(vector_store)
print('load_Imdb3')
load_Imdb3(vector_store)



