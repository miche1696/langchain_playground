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


from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

hyde_template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_template)

hyde_doc_generated = hyde_prompt | ChatOpenAI(temperature=0) | StrOutputParser()

# Prompt
question = "What is task decomposition for LLM agents?"

# Retriever
retrieval_chain = hyde_doc_generated | retriever 
retrieved_docs = retrieval_chain.invoke({"question":question})

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context": retrieved_docs,"question":question})