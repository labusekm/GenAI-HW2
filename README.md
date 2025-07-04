# GenAI-HW4
RAG-based movies / tv shows extraction

This project is a module of larger latforms content - recommendation solution, which has the following modules:
- recognition of current movie / tv show feedback or reaction on specific path of content
- usage of this module for looking for further movie / tv show recommendation based on user feedback and profile (eq. age of user)
- above result can be used for next content recommendation or fedback for platform content management team

## High level view
![HighLevel](https://github.com/user-attachments/assets/21e0db2c-fbfe-4c11-8cc0-e4b14eacf9eb)


Solution contains csv data export from various sources and contain specific platform content and data from imdb. Sum of records is ~30 000.
Solution uses RAG to retrieve specific and precise data and avoid halucination when passing above requirements.
- Data is extracted to csv formant to allow further processing (Cut CSV)
- Movie posters are strictly connected with movies and they present what movie can give to the viewers. They also share emotions viewers can feel during watching. Hence LLM is asked to explain each poster and this information is provided to vector DB (LLM: text metadata)

RAG data are retrieved using "Retriever". Retriever is also responsible for decoration of documents with relevant documents, based on "title" and "imdb_id" fields in metadata (if existing). This functionality is not required to be executed using LLM - it saves cost of LLM and enhances security as agents do only what is required

React agent reads this data and checks what is missing in RAG. Missing data are requested using tool (tmdb tool) which calls tmdb API. tmdb API is always requested for newest vote of  amovie, to retrieve up to date data

All the result is provided to "extract agent" to generate content from all the data available.

The following decisions were made in application design
- modularity - task based agents and other tasks can be implemented without LLMs - it saves cost, enhances maintainability and security of LLM usage
- each embedding chunk is wole record from input data - it's enough for good quality and cost effective solution (solution is still efficient when invoking llm with param: max_tokens=50)
- added title, type, age and platform availability into metadata to improve the speed of data filtering
- used HuggingFace embedding as recommended fine-tuned solution
- Cohere "command" model is recommended for RAG-oriented tasks, Gpt-4o for image processing and agents - more powerful moel
- used search_kwargs parameter for efficient data filtering (eq. prefiltering of movies recommended for specific age of user)


How to use the solution
1. clone the repository
2. run pip install
3. create api hey on Cohere
4. Add api key to environment variables
5. run py load.py to generate embeddings in vector storage (~3-4 hours)
6. run py extract.py to check two approaches of data extraction
