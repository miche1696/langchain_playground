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



#### INDEXING ####

# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate

## Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union

docs = retrieval_chain.invoke({"question":question})

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough


## RAG
from langchain import hub
prompt_hub_rag = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(temperature=0)

multiquery_chain_query_translation = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt_hub_rag
    | llm
    | StrOutputParser()
)

multiquery_chain_query_translation.invoke({"question":question})