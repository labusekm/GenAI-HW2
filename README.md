# GenAI-HW2
RAG-based movies / tv shows extraction

This project is a module of larger latforms content - recommendation solution, which has the following modules:
- recognition of current movie / tv show feedback or reaction on specific path of content
- usage of this module for looking for further movie / tv show recommendation based on user feedback and profile (eq. age of user)
- above result can be used for next content recommendation or fedback for platform content management team

Solution contains csv data export from various sources and contain specific platform content and data from imdb. Sum of records is ~30 000.
Solution uses RAG to retrieve specific and precise data and avoid halucination when passing above requirements.
Data is extracted to csv formant to allow further processing.

The following decisions were made in application design
- each embedding chunk is wole record from input data - it's enough for good quality and cost effective solution
- added title, type, age and platform availability into metadata to improve the speed of data filtering
- used HuggingFace embedding as recommended fine-tuned solution
- Cohere "command" model is recommended for RAG-oriented tasks
- used search_kwargs parameter for efficient data filtering (eq. prefiltering of movies recommended for specific age of user)
- as required data is unstructured and spreaded across various input document, specific prompt was required to match records by title: "Match two documents from embeddings by same title if it''s required." As a result, the following records from embeddings were matched:
  ![documents](https://github.com/user-attachments/assets/58fef2a5-aec8-4895-be30-8db3aab8ee75)
  Results show match was properly made based on the following query (taken from overview of above embedding):  "Find american movies about robot who look like kid. Actor staring in movie should be famous."
  title,maturity,genre,platforms\n
  "A.I. Artificial Intelligence",13+,American,Prime Video\n


How to use the solution
1. clone the repository
2. run pip install
3. create api hey on Cohere
4. Add api key to environment variables
5. run py load.py to generate embeddings in vector storage (~3-4 hours)
6. run py extract.py to check two approaches of data extraction
