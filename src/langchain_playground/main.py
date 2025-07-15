import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# Set LangChain API key from environment variable
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
if langchain_api_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
else:
    print("Warning: LANGCHAIN_API_KEY not found in environment variables")

# Set OpenAI API key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
else:
    print("Warning: OPENAI_API_KEY not found in environment variables")

# Set User Agent for web scraping
user_agent = os.getenv('USER_AGENT')
if user_agent:
    os.environ['USER_AGENT'] = user_agent
else:
    print("Warning: USER_AGENT not found in environment variables")

if __name__ == "__main__":
    # Import and run the .py script
    import sys
    import os
    
    # Add the chains directory to the path
    chains_dir = os.path.join(os.path.dirname(__file__), 'chains')
    sys.path.insert(0, chains_dir)
    
    # Import and execute the .py script inside "chains"
    try:
        import multiquery_query_translation
        print("Successfully executed .py")
    except ImportError as e:
        print(f"Error importing .py: {e}")
    except Exception as e:
        print(f"Error running .py: {e}")