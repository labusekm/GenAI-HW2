from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

persist_directory = "data/vectorstore"
embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

#query data with "hero" keyword and age less than 18 years old
query = "Find american movies about robot who look like kid. Actor staring in movie should be famous. Do not repeat similar titles"
records = vectordb.similarity_search(query,k=5)

print(records)