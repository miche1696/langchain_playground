import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

langchain_project = os.getenv('LANGCHAIN_PROJECT')
if langchain_project:
    os.environ['LANGCHAIN_PROJECT'] = langchain_project

# Set LangChain API key from environment variable
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
if langchain_api_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
else:
    print("Warning: LANGCHAIN_API_KEY not found in environment variables")

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
else:
    print("Warning: OPENAI_API_KEY not found in environment variables")

import os

from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch


file = './resources/OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

from langchain.indexes import VectorstoreIndexCreator

embedding = OpenAIEmbeddings()
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embedding
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",  # cheap, classical completion model
    temperature=0
)

response = index.query(query, llm)
# print(response)

from rich.console import Console
from rich.markdown import Markdown

console = Console()
#console.print(Markdown(response))

docs = loader.load()

#print(docs[0])

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

"""
embed = embeddings.embed_query("Hi my name is Harrison")
print(len(embed))
print(embed[:5])
"""

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

query = "Please suggest a shirt with sunblocking"

"""
docs = db.similarity_search(query)
print(len(docs))
print(docs[0])
"""

retriever = db.as_retriever()

""" # Doing it manually 

llm_model = OpenAI(
    model="gpt-3.5-turbo",  # cheap, classical completion model
    temperature=0
)

llm = ChatOpenAI(temperature = 0.0, model=llm_model)

qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

console.print(Markdown(response))

"""

""" # Done with a chain / RetrievalQA

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)

console.print(Markdown(response))

"""