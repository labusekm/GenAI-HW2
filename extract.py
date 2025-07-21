from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Cohere

from langgraph.prebuilt import create_react_agent
from langchain.agents import load_tools
from langchain_openai import AzureChatOpenAI
import os
import logging
import uuid

new_uuid = uuid.uuid4()
uuid_string = str(new_uuid)

logging.basicConfig(
    filename='debug.log',  # Specify the log file name
    level=logging.DEBUG,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log message format
)

persist_directory = "data/vectorstore"
embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

def extract_titles(query, command, filter = None):
    if filter is not None:
        retriever = vectordb.as_retriever(search_kwargs={"k": 5, "filter": filter})
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})       

    documents = query_vector_store(query, retriever)
    if documents:
        response = movie_details_agent(documents, command)
        return extract_llm(response, command)
    else:
        return "No relevant documents found."
    

    

def query_vector_store(query, retriever):
    print('retriever started')
    documents = retriever.invoke(query)
    logging.info(f"{uuid_string}: Documents retrieved: {documents}")
    #search documents by relation (in metadata)
    decorate_documents_with_relations(documents)
    logging.info(f"{uuid_string}: Documents retrieved: {documents}")
    print('documents retrieved from RAG')
    return documents


def decorate_documents_with_relations(documents):
    for doc in documents:
        if 'imdb_id' in doc.metadata:
            logging.info(f"{uuid_string}: searching for imdb_id entries: {doc.metadata['imdb_id']}")
            relevant_documents = vectordb.get(where={"imdb_id": doc.metadata['imdb_id']})
            metadatas = relevant_documents['metadatas']
            logging.info(f"{uuid_string}: {len(metadatas)} documents found.")
            logging.info(f"{uuid_string}: relevant_documents: {relevant_documents}")
            index_to_exclude = relevant_documents['ids'].index(doc.id)               
            for i, metadata in enumerate(relevant_documents['metadatas']):
                if i != index_to_exclude:
                    if 'title' in metadata:
                        logging.info(f"{uuid_string}: searching for title entries: {metadata['title']}")
                        relevant_docs_title = vectordb.get(where={"title": metadata['title']})
                        logging.info(f"{uuid_string}: {len(relevant_docs_title)} documents found by title.")
                        logging.info(f"{uuid_string}: documents found by title: {relevant_docs_title}")
                        doc.page_content = doc.page_content + str(relevant_docs_title["documents"])
                    else:
                        filtered_documents = exclude_duplicate_document_used_for_search(relevant_documents, index_to_exclude)   
                        doc.page_content = doc.page_content + str(filtered_documents) 
                        
        logging.info(f"{uuid_string}: Document decorated with relations: {doc}.")

def exclude_duplicate_document_used_for_search(relevant_documents, index_to_exclude):
    filtered_documents = [
        doc for i, doc in enumerate(relevant_documents['documents'])
        if i != index_to_exclude
    ]
    return filtered_documents

def extract_llm(documents, command):   
    #TODO: will work when movie_details_agent will work
    context = "\n".join([doc for doc in documents])
    myprompt = f"Given the context:\n{context}\n Perform command: {command} for records from context"
    logging.info(f"{uuid_string}: prompt: {myprompt}")
    COHERE_API_KEY = os.environ["COHERE_API_KEY"]
    llm=Cohere(model='command',cohere_api_key=COHERE_API_KEY)
    response = llm.generate(prompts=[myprompt])
    logging.info(f"{uuid_string}: llm generation: {response}")
    return response

def movie_details_agent(documents, command):    
    DIAL_API_KEY = os.environ["DIAL_API_KEY"]
    model = AzureChatOpenAI(
        api_version="2024-10-21",
        azure_endpoint="https://ai-proxy.lab.epam.com",
        api_key=DIAL_API_KEY,
        model="gpt-4o")
    tmdb_bearer_token = os.environ["TMDB_API_KEY"]
    tools = load_tools(["tmdb-api"], llm=model, tmdb_bearer_token=tmdb_bearer_token)
    response_docs = []
    for doc in documents:
        myprompt = f"""You are a movie advisor, you answer questions about movies. 
                                        Given the context as list of documents:\n{doc}\n, check if there is enough information to perform a command: {command} for records from context
                                        If you don't know an answer, invoke the TMDB-API with a question in natural language. 
                                        If the command contains movie vote or movie mark, always invoke TMDB-API for this field.
                                        Return with same context and replace missing data with response from TMDB-API.
                                        Only the URL of the end point should be returned and no extra text explaining why the URL was built the way it was.
                                        """
        print("started react agent")

        agent_executor  = create_react_agent(
            model,
            tools)
        
        response = agent_executor.invoke({"messages": [{"role": "user", "content": myprompt}]})
        logging.info(f"{uuid_string}: react agent response: {response}")
        print("react agent end")
        # Take last AIMessage in Chain Of Thought output
        messages = response['messages']

        response = messages[len(messages)-1]

        response_docs.append(response.content)

    return response_docs
#####################################################################
# Extract top 5 titles matching the query

# result = extract_titles('Find american movies about robot who look like kid. Actor staring in movie should be famous. Do not repeat similar titles. Match two documents from embeddings by same title if it''s required.', 
#                             'Create comma delimited csv data with columns: title, maturity rating, genre, list of streaming platforms it is avalaible on, of all records matching the query. \n ' \
# 'Format of csv header should be: title, maturity, genre, platforms.Include header of csv\n')
# print(result)

#####################################################################
# Extract top 5 titles also suitable for 13 - year old child

# filter = {"age": {'$lte': 13}}
# result = extract_titles('Find american movies about robot who look like kid. Actor staring in movie should be famous. Do not repeat similar titles. Match two documents from embeddings by same title if it''s required.', 
#                             'Create comma delimited csv data with columns: title, maturity rating, genre, list of streaming platforms it is avalaible on, of all records matching the query. \n ' \
# 'Format of csv header should be: title, maturity, genre, platforms.Include header of csv\n', filter)

# print(result)


#################################### task 4 ##########################
result = extract_titles('Find exactly 5 poster descriptions which make the viewer interested in the future relation between people and technology. It should touch machine feelings and problems with integration with human. Viewer should feel emotional connection with both character types.', 
                            'Create comma delimited csv data with records and columns: title, vote, genre, list of streaming platforms it is avalaible. Do not make up an answer, use only context provided\n ' \
'Format of csv header should be: title, vote, genre, platforms.Include header of csv\n Do not respond with anything but raw csv content')
print(result)