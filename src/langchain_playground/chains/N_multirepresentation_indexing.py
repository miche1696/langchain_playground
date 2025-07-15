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

# --- Imports for Langchain components ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid

# --- Load Documents ---
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())

print(docs)

# --- Assign unique IDs to each document ---
doc_ids = [str(uuid.uuid4()) for _ in docs]

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOpenAI(model="gpt-3.5-turbo",max_retries=0)
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 5})

from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# --- Multi-Representation Indexing with Summaries ---

# 1. Create a Chroma vectorstore for summaries
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())

# 2. Create a docstore for full documents
store = InMemoryByteStore()
id_key = "doc_id"

# 3. Prepare summary Documents with shared doc_id
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# 4. Initialize MultiVectorRetriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# 5. Add summaries to vectorstore and full docs to docstore
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# --- Example usage ---
# Querying the retriever returns the full document(s) associated with the best-matching summary
query = "data quality"
retrieved_docs = retriever.invoke(query, config={"run_id": "MySpecialQuery"})
# print("Retrieved document content (first 500 chars):\n", retrieved_docs[0].page_content[:500])
print("Metadata:", retrieved_docs[0].metadata)