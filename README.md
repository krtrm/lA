

LegalEase.app is an AI-powered legal assistant specifically designed for the Indian legal context. It leverages advanced language models and a sophisticated Retrieval-Augmented Generation (RAG) pipeline to provide assistance with legal research, document analysis, and drafting tasks related to Indian law.




## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [RAG Pipeline Explained](#rag-pipeline-explained)
  - [Core Concept](#core-concept)
  - [Knowledge Base Creation (Offline)](#knowledge-base-creation-offline)
  - [Query Processing (Online)](#query-processing-online)
  - [Specialized Functions](#specialized-functions)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

LegalEase.app offers several specialized tools for legal professionals and students:

1.  **Legal Keyword Extraction:** Analyzes legal text to identify and define key legal terms specific to Indian law, aiding comprehension and study. Uses LLMs to understand context and provide relevant definitions.
2.  **Legal Argument Composer:** Generates structured legal arguments based on a given topic and key points. It retrieves relevant legal principles and precedents from the knowledge base and web search to build a coherent and persuasive argument.
3.  **Document Outline Generator:** Creates professional outlines for various legal documents (e.g., briefs, memos, contracts, petitions, complaints, wills) based on standard Indian legal practices and the specific topic provided.
4.  **Citation Verifier:** Checks the validity and format of Indian legal citations (cases, statutes). It uses the Indian Kanoon API and vector database lookups to confirm existence, provides corrections based on standard formats (like SCC, AIR), and offers brief summaries of the cited material.
5.  **Legal Q&A (Implicit):** The underlying RAG system can answer specific questions about Indian law by retrieving relevant information from its internal knowledge base (indexed legal documents) and performing targeted web searches for the most current information.

## Technology Stack

-   **Backend Framework:** Python with FastAPI for building the robust API, Uvicorn as the ASGI server.
-   **Frontend Framework:** Streamlit for creating the interactive user interface.
-   **AI/LLM Orchestration:** Langchain for managing the RAG pipeline, prompts, and interactions with different models.
-   **Language Models (LLMs):**
    *   OpenAI API: Utilized for powerful models like GPT-4 variants (e.g., `gpt-4.1-mini`) for generation and `text-embedding-3-small` for creating embeddings.
    *   Groq API: Leveraged for fast inference with models like Llama 3 and Mixtral (e.g., `deepseek-r1-distill-llama-70b`) for planning, evaluation, and potentially generation.
-   **Vector Database:** Pinecone for efficient storage and retrieval of document embeddings.
-   **Data Processing & Loading:**
    *   `PyMuPDF` (Fitz): For extracting text from PDF documents.
    *   `Pytesseract`: OCR engine for extracting text from image-based PDFs or images.
    *   `BeautifulSoup4` & `Html2Text`: For parsing and cleaning HTML content from web pages.
    *   `Unstructured`: For loading and processing various document formats (DOCX, etc.).
    *   `PyTube`: For fetching YouTube video transcripts.
    *   `python-docx`: For handling .docx files.
    *   `lxml`: Efficient XML and HTML parsing.
-   **Search Tools:**
    *   `ikapi` (Custom Wrapper): Interacts with the Indian Kanoon API for searching specific Indian case law and judgments.
    *   Serper API: Provides Google Search results, focused on Indian region (`gl='in'`) for web search capabilities.
-   **Database (Optional/Future):** SQLite managed via `aiosqlite` and potentially `SQLAlchemy` for storing user data, chat history, or other relational data.
-   **Other Key Libraries:** `python-dotenv` (environment variables), `httpx` / `requests` (HTTP requests), `tiktoken` (token counting), `Pillow` (image processing for OCR).

## RAG Pipeline Explained

### Core Concept

Retrieval-Augmented Generation (RAG) combines the strengths of large language models (LLMs) with external knowledge retrieval. Instead of relying solely on the LLM's pre-trained (and potentially outdated) knowledge, RAG first retrieves relevant information from a specific knowledge base and then provides this information as context to the LLM when generating a response. This makes the output more accurate, up-to-date, and grounded in factual data, which is crucial for the legal domain.

### Knowledge Base Creation (Offline)

This is the process of preparing the specialized knowledge source:

1.  **Document Loading:** Various legal documents relevant to Indian law (PDFs, DOCX, potentially web scrapes of specific legal sites) are collected. The `backend/books/` directory is prioritized for core legal texts.
2.  **Text Extraction:** Text is extracted from these documents. For PDFs, `PyMuPDF` is used. If a PDF contains images or scanned text, `Pytesseract` performs Optical Character Recognition (OCR) to convert images to text.
3.  **Chunking:** The extracted text is divided into smaller, manageable chunks (e.g., 1000 characters with 200 overlap) using `RecursiveCharacterTextSplitter`. This ensures that semantic context is preserved within chunks and allows for efficient retrieval.
4.  **Embedding:** Each text chunk is converted into a high-dimensional numerical vector (embedding) using an embedding model (e.g., OpenAI's `text-embedding-3-small`). These vectors capture the semantic meaning of the text. The embeddings are sliced to 1024 dimensions using the custom `SlicedOpenAIEmbeddings` class to match the configuration of the Pinecone index.
5.  **Indexing:** The text chunks and their corresponding embeddings are stored in a Pinecone vector database. Pinecone allows for efficient similarity searches, enabling the system to quickly find text chunks whose meanings are closest to a user's query.

### Query Processing (Online)

This happens in real-time when a user interacts with the application:

1.  **Input:** A user submits a query or text via the Streamlit frontend.
2.  **Tool Necessity Check:** A lightweight, fast LLM (potentially via Groq) quickly assesses if the query is simple conversational chat or requires factual information retrieval using tools. Simple chat might be handled directly by an LLM without retrieval.
3.  **Planning (If Tools Needed):** For complex queries requiring external data, a more capable planning LLM (e.g., a model on Groq like `deepseek-r1`) analyzes the query and devises a multi-step plan. This plan outlines which tools (Vector DB Lookup, Indian Kanoon Search, Web Search) should be used and in what order. The plan prioritizes searching the internal Pinecone knowledge base first.
4.  **Tool Execution (Parallel/Sequential):** The `ToolManager` executes the plan:
    *   **Vector DB Lookup (`VectorDBLookupTool`):** The user's query is embedded, and Pinecone is searched to find the most semantically similar text chunks from the indexed legal documents. This is the primary source for grounding answers in the curated knowledge base.
    *   **Indian Kanoon Search (`IKAPITool`):** If the query involves specific case names, citations, or requires searching judgments, this tool queries the Indian Kanoon database via its API. Results are cached locally in the `ik_data/` directory.
    *   **Web Search (`WebSearchTool`):** If the internal knowledge base and Indian Kanoon are insufficient, or if very recent information is needed, this tool queries the web using the Serper API (focused on India). It attempts to extract content from various sources (HTML pages, PDFs, YouTube transcripts) and prioritizes reliable domains (gov.in, legal sites).
5.  **Context Augmentation & Ranking:** Information retrieved from all executed tools is collected. The system ranks these pieces of information based on relevance to the query (e.g., using vector similarity scores) and filters them to fit within the LLM's context window token limit (e.g., using `_select_docs_within_budget`).
6.  **Generation:** The original query and the curated, augmented context (retrieved information) are passed to a powerful generator LLM (e.g., `gpt-4.1-mini` or a large Groq model).
7.  **Prompt Engineering:** Carefully crafted prompts instruct the LLM to act as an Indian legal expert, synthesize the provided context, answer the user's query accurately, cite sources appropriately (using Indian legal citation standards), and format the output as required (e.g., JSON for keywords, Markdown for arguments/outlines).
8.  **Response & Streaming:** The LLM generates the final response. For interactive features, the backend uses Server-Sent Events (SSE) over `ndjson` (newline-delimited JSON) to stream intermediate steps (Thinking, Planning, Tool Use, Retrieval, Generating) and the final result to the Streamlit frontend, providing real-time feedback to the user.

### Specialized Functions

The specific features like Keyword Extraction, Argument Composer, Outline Generator, and Citation Verifier use variations of this RAG pipeline, often with tailored prompts and potentially specific tool usage patterns (e.g., Citation Verifier heavily relies on `IKAPITool` and vector search for context).

## Setup Instructions

Follow these steps to set up and run LegalEase.app locally.

### Prerequisites

-   **Python:** Version 3.9 or higher. ([Download Python](https://www.python.org/downloads/))
-   **pip:** Python package installer (usually comes with Python).
-   **Git:** For cloning the repository. ([Download Git](https://git-scm.com/downloads))
-   **Tesseract OCR:** Required for extracting text from image-based PDFs. Installation varies by OS:
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install tesseract-ocr libtesseract-dev`
    *   **macOS (using Homebrew):** `brew install tesseract`
    *   **Windows:** Download the installer from the [official Tesseract repository](https://github.com/UB-Mannheim/tesseract/wiki). **Crucially, ensure `tesseract.exe` is added to your system's PATH environment variable during or after installation.**

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/a3ro-dev/LegalEase 
    cd LegalEase # Or your repository's root directory name
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows (Command Prompt/PowerShell):**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *(Your terminal prompt should now be prefixed with `(venv)`)*

3.  **Install Dependencies:**
    Install all required Python packages listed in the `requirements.txt` file located in the `backend` directory.
    ```bash
    pip install -r backend/requirements.txt
    ```
    *(Note: This single file includes dependencies for both the FastAPI backend and the Streamlit frontend.)*

### Environment Variables

Sensitive information like API keys are managed using environment variables.

1.  **Create a `.env` file:** In the project's root directory, create a file named `.env`.

2.  **Populate `.env`:** Copy the template below into your `.env` file. **Replace the placeholder values (`"YOUR_..._KEY"`, `"your-..."`) with your actual API keys and settings.**

    ```dotenv
    # .env.template - Copy this to .env and fill in your values

    # --- API Keys ---
    # Get from https://platform.openai.com/api-keys
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    # Get from https://console.groq.com/keys
    GROQ_API_KEY="YOUR_GROQ_API_KEY"
    # Get from https://app.pinecone.io/
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    # Get from https://indiankanoon.org/api/register/ (May require approval & has usage limits)
    INDIANKANOON_API_TOKEN="YOUR_INDIANKANOON_API_TOKEN"
    # Get from https://serper.dev/ (Provides Google Search results)
    SERPER_API_KEY="YOUR_SERPER_API_KEY"
    # Get from https://newsapi.ai/ (Alternative news source if needed)
    # NEWS_API_KEY="YOUR_NEWSAPI_AI_KEY"

    # --- Pinecone Settings ---
    # Your Pinecone index name. MUST exist and match embedding dimensions (1024).
    PINECONE_INDEX_NAME="your-pinecone-index-name" # e.g., "legal-india-index"
    # Namespace within the index (optional, good for organizing data)
    PINECONE_NAMESPACE="your-pinecone-namespace" # e.g., "indian-law-docs"
    # Pinecone environment/region (e.g., gcp-starter, us-east-1-aws, etc.)
    # Check your Pinecone console for the correct value.
    PINECONE_REGION="gcp-starter"

    # --- Embeddings Model ---
    # Model used for creating embeddings (Sliced to 1024 dimensions in code)
    EMBEDDINGS_MODEL="text-embedding-3-small"

    # --- Application Settings ---
    # Set to "production" or "development". Controls HTTPS redirect, debug modes.
    ENVIRONMENT="development"
    # URL for the FastAPI backend (used by Streamlit frontend)
    API_URL="http://localhost:8000"

    # --- Data Directories (Optional - Defaults are usually fine) ---
    # Directory to store data fetched by Indian Kanoon API
    # Default: ik_data/ relative to project root
    # INDIANKANOON_DATA_DIR="/path/to/your/ik_data"
    ```

3.  **Explanation & Verification:**
    *   Ensure you have accounts and valid API keys for OpenAI, Groq, Pinecone, Indian Kanoon (if approved), and Serper.
    *   **Crucially:** The `PINECONE_INDEX_NAME` must correspond to an existing index in your Pinecone account, and that index must be configured for 1024 dimensions to match the `SlicedOpenAIEmbeddings` output.
    *   The `PINECONE_REGION` must match the region where your index is hosted.
    *   Keep the `.env` file secure and **do not commit it to version control** (ensure `.env` is listed in your `.gitignore` file).

### Running the Application

The application consists of two parts: the backend API (FastAPI) and the frontend UI (Streamlit). They need to be run separately, typically in two different terminal windows/tabs. **Ensure your virtual environment is activated in both terminals.**

1.  **Run the Backend (FastAPI):**
    Navigate to the `backend` directory in your terminal and start the Uvicorn server:
    ```bash
    cd backend
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    ```
    *   `api:app`: Tells Uvicorn to load the `app` object from the `api.py` file.
    *   `--host 0.0.0.0`: Makes the server accessible from your local machine and potentially other devices on your network.
    *   `--port 8000`: Runs the server on port 8000 (matching the default `API_URL` in `.env`).
    *   `--reload`: Enables auto-reload. Uvicorn watches for code changes and restarts the server automatically (very useful during development).

    Wait for output indicating the server is running and the RAG system is initialized, like:
    ```
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    INFO:     Started reloader process [...] using statreload
    INFO:     Started server process [...] 
    INFO:     Waiting for application startup.
    INFO:     Attempting to initialize RAG system with index: 'your-pinecone-index-name', namespace: 'your-pinecone-namespace'
    INFO:     RAG system initialized successfully
    INFO:     Application startup complete.
    ```
    *(If you see errors here, double-check your `.env` file, Pinecone index configuration, and API keys.)*

2.  **Run the Frontend (Streamlit):**
    Open a **new terminal window/tab**, activate the virtual environment (`source venv/bin/activate` or `.\venv\Scripts\activate`), navigate to the `frontend` directory, and run the Streamlit app:
    ```bash
    cd ../frontend # Navigate back to root, then into frontend
    streamlit run app.py
    ```

    Streamlit will start its server (usually on port 8501) and should automatically open the application in your default web browser. The terminal output will provide the URLs:
    ```
    You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://<your-local-ip>:8501
    ```

### Testing the Dockerized Setup

Once both services are defined in `docker-compose.yml`, you can verify they run together and serve requests:

1. Build and start in detached mode  
   ```bash
   docker-compose up --build -d
   ```
2. Confirm both containers are healthy and listening  
   ```bash
   docker-compose ps
   ```
   You should see `backend` on port 8000 and `frontend` on port 8501.

3. Inspect logs for errors or startup messages  
   ```bash
   docker-compose logs -f
   ```

4. Test the FastAPI health endpoint  
   ```bash
   curl http://localhost:8000/
   ```
   Expect a JSON welcome message:
   ```json
   {"message":"Welcome to the Vaqeel.app Legal AI API"}
   ```

5. Test a simple API query  
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query":"What is Section 420 IPC?","use_web":false}'
   ```

6. Verify the Streamlit frontend  
   - Open your browser at http://localhost:8501  
   - The LegalEase UI should load without errors, and sidebar functions should call the API.

7. Tear down the setup  
   ```bash
   docker-compose down
   ```

## Usage

1.  Ensure both the backend (Uvicorn) and frontend (Streamlit) servers are running as described above.
2.  Open the Streamlit application URL in your web browser (typically `http://localhost:8501`).
3.  Use the sidebar on the left to navigate between the different functions offered by LegalEase:
    *   Legal Keyword Extraction
    *   Legal Argument Composer
    *   Document Outline Generator
    *   Citation Verifier
4.  Within each function's page, follow the specific instructions provided.
    *   Input text, topics, key points, or citations as required.
    *   Use the expander sections (`ℹ️ How to use this feature`) for guidance.
5.  Click the primary action button (e.g., "Extract Keywords", "Generate Argument", "Generate Outline", "Verify Citation").
6.  Wait for the AI to process the request. The interface will show status updates (Thinking, Retrieving, Generating...) thanks to the streaming responses from the backend.
7.  Review the results displayed in the main area. Download options (e.g., JSON, Markdown) are available for most outputs.

## Contributing

We welcome contributions to LegalEase.app! Whether it's reporting bugs, suggesting features, or submitting code changes, your help is appreciated.

Please refer to our [**CONTRIBUTING.md**](CONTRIBUTING.md) file for detailed guidelines on:

*   Reporting issues
*   Suggesting enhancements
*   Setting up your development environment
*   Coding standards
*   Submitting Pull Requests

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. *(Ensure a LICENSE file with the MIT license text exists in the repository root)*





