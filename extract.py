from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Cohere
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, load_tools
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import os
import logging
import uuid
from langchain_openai import AzureChatOpenAI

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
        

    COHERE_API_KEY = os.environ["COHERE_API_KEY"]
    llm=Cohere(model='command',cohere_api_key=COHERE_API_KEY)

    # documents = query_vector_store(query, retriever)
    # if documents:
    #     extraction = extract_llm(documents, llm, command)
    # else:
    #     extraction = "No relevant documents found."
    test_result = agent_llm()
    return test_result

def query_vector_store(query, retriever):
    print('retriever started')
    documents = retriever.invoke(query)
    #search documents by relation (in metadata)
    decorate_documents_with_relations(documents)
    logging.info(f"{uuid_string}: Documents retrieved: {documents}")
    return documents

def extract_llm(documents, llm, command):   
    context = "\n".join([doc.page_content for doc in documents])
    myprompt = f"Given the context:\n{context}\n Perform command: {command} for records from context"
    response = llm.generate(prompts=[myprompt])
    logging.info(f"{uuid_string}: llm generation: {response}")
    return response

def agent_llm():
    DIAL_API_KEY = os.environ["DIAL_API_KEY"]
    llm = AzureChatOpenAI(
        api_version="2024-10-21",
        azure_endpoint="https://ai-proxy.lab.epam.com",
        api_key=DIAL_API_KEY,
        model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie advisor, you answer questions about movies."),
        ("system", "If you don't know an answer, invoke the TMDB-API with a question in natural language."),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    tmdb_bearer_token = os.getenv("TMDB_BEARER_TOKEN")
    tools = load_tools(["tmdb-api"], llm=llm, tmdb_bearer_token=tmdb_bearer_token)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke(
    {
        "query": "Give me the plot of movie 'Inception'?",
    })
    return result


def decorate_documents_with_relations(documents):
    for doc in documents:
        if 'imdb_id' in doc.metadata:
            logging.info(f"{uuid_string}: searching for imdb_id entries: {doc.metadata['imdb_id']}")
            relevant_documents = vectordb.get(where={"imdb_id": doc.metadata['imdb_id']})
            metadatas = relevant_documents['metadatas']
            logging.info(f"{uuid_string}: {len(metadatas)} documents found.")
            doc.page_content = doc.page_content + str(relevant_documents['documents'])
            for metadata in metadatas:
                print(metadata)
                if 'title' in metadata:
                    logging.info(f"{uuid_string}: searching for title entries: {metadata['title']}")
                    relevant_docs_title = vectordb.get(where={"title": metadata['title']})
                    logging.info(f"{uuid_string}: {len(relevant_docs_title)} documents found by title.")
                    doc.page_content = doc.page_content + str(relevant_docs_title)
    
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
                            'For all records matching the query, create comma delimited csv data with records and columns: title, maturity rating, genre, list of streaming platforms it is avalaible. Do not make up an answer, use only context provided\n ' \
'Format of csv header should be: title, maturity, genre, platforms.Include header of csv\n Do not respond with anything but raw csv content')
print(result)