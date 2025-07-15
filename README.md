# LangChain Playground

Test, study, and experimentation with LangChain and LangGraph.

## Setup

### Environment Variables

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Edit the `.env` file and add your API keys:
   ```bash
   # LangChain API Configuration
   LANGCHAIN_API_KEY=your_actual_langchain_api_key_here
   
   # Other environment variables can be added here
   # OPENAI_API_KEY=your_openai_api_key_here
   ```

### Installation

Install dependencies using Poetry:
```bash
poetry install
```

## Usage

Run the main application:
```bash
poetry run python src/langchain_playground/main.py
```
