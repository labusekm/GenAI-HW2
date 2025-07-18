from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Cohere
from langchain_experimental.tools import PythonAstREPLTool
import os

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

    return query_vector_store(query, command, retriever, llm)


def query_vector_store(query, command, retriever, llm):
    documents = retriever.invoke(query)
    
    if documents:
        context = "\n".join([doc.page_content for doc in documents])
        myprompt = f"Given the context:\n{context}\n Perform command: {command} for records matching the following query: {query}"
        response = llm.generate(prompts=[myprompt])
        return response
    else:
        return "No relevant documents found."
    
#####################################################################
# Extract top 5 titles matching the query

result = extract_titles('Find american movies about robot who look like kid. Actor staring in movie should be famous. Do not repeat similar titles. Match two documents from embeddings by same title if it''s required.', 
                            'Create comma delimited csv data with columns: title, maturity rating, genre, list of streaming platforms it is avalaible on, of all records matching the query. \n ' \
'Format of csv header should be: title, maturity, genre, platforms.Include header of csv\n')
print(result)

#####################################################################
# Extract top 5 titles also suitable for 13 - year old child

filter = {"age": {'$lte': 13}}
result = extract_titles('Find american movies about robot who look like kid. Actor staring in movie should be famous. Do not repeat similar titles. Match two documents from embeddings by same title if it''s required.', 
                            'Create comma delimited csv data with columns: title, maturity rating, genre, list of streaming platforms it is avalaible on, of all records matching the query. \n ' \
'Format of csv header should be: title, maturity, genre, platforms.Include header of csv\n', filter)

print(result)