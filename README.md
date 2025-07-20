# Generic MCP Server for Data Analysis

This project provides a flexible and powerful Model Context Protocol (MCP) server designed to integrate with Large Language Models (LLMs). It serves as a template for building applications that can understand user queries, retrieve relevant data from various sources, process it, and provide it to an LLM for analysis and response generation.

## Core Features

- **Query Analysis**: Interprets natural language user requests to understand intent.
- **Dynamic Data Retrieval**: Selects and fetches relevant data files (e.g., Excel, CSV) from a predefined collection based on the user's query.
- **Data Processing**: Converts raw data files into a structured format like JSON, making them easy for LLMs to consume.
- **LLM Integration**: Seamlessly provides structured data to an LLM for in-depth analysis, summarization, or answering questions.
- **Extensible**: Easily adaptable to different domains and datasets beyond the initial example.

## How It Works: A High-Level Workflow

1.  **User Query**: A user submits a question or a request in natural language.
2.  **Analysis & File Selection**: The MCP server analyzes the query to identify key topics and intents. It then intelligently selects the most relevant data files from a local or remote repository.
3.  **Data Retrieval & Conversion**: The server downloads the selected files and converts them from their original format (e.g., Excel) into clean, structured JSON.
4.  **LLM-Ready Output**: The processed data, along with metadata and a summary, is prepared for an LLM. This output can be directly fed into a model like GPT, Claude, or Gemini.
5.  **Insight Generation**: The LLM uses the provided data to generate a comprehensive, data-driven response to the user's original query.

## Getting Started

### Prerequisites

- Python 3.8+
- An LLM API key (e.g., OpenAI, Anthropic, Google AI) for the final analysis step.

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare your data**:
    - Place your data files (e.g., `.xls`, `.xlsx`, `.csv`) in a designated directory (e.g., `data/`).
    - Create a `data.json` file that lists your available datasets and their metadata. This file is crucial for the file selection logic. See the existing `data.json` for an example structure.

### Running the Server

To start the MCP server, run:

```bash
python server.py
```

By default, the server will be accessible at `http://0.0.0.0:8050`.

## Server Tools (API)

The server exposes functions that can be called by an MCP client or an LLM agent.

### `analyze_and_select_data(user_question: str)`

-   **Purpose**: To analyze the user's question and identify the most relevant data files.
-   **Input**: A string containing the user's question.
-   **Output**: A JSON object detailing the selected files and the reasoning behind the selection.

### `retrieve_and_process_data(selected_files_json: str)`

-   **Purpose**: To download or read the selected files and convert them into a structured format.
-   **Input**: The JSON output from the `analyze_and_select_data` function.
-   **Output**: A JSON object containing the processed data, ready for LLM consumption.

## Customization

This server is a template. You can customize it for your specific needs:

-   **Data Sources**: Modify the `data.json` and the data retrieval logic to work with your own datasets, whether they are local files or accessible via an API.
-   **File Selection Logic**: Enhance the keyword matching or implement more advanced NLP techniques in `analyze_and_select_data` to improve the relevance of file selection for your domain.
-   **Data Processing**: Add support for other file formats (e.g., CSV, Parquet) or implement custom data cleaning and transformation pipelines in the `retrieve_and_process_data` function.

## Dependencies

-   `fastapi`: For creating the web server.
-   `uvicorn`: For running the FastAPI application.
-   `pandas`: For data manipulation and reading Excel/CSV files.
-   `openpyxl`: Required by pandas for `.xlsx` files.
-   `xlrd`: Required by pandas for legacy `.xls` files.
-   `python-multipart`: For handling file uploads.
-   `requests`: For making HTTP requests to download files.

## Project Structure

```
.
├── server.py             # The main MCP server application.
├── data.json             # JSON file with metadata about your datasets.
├── requirements.txt      # Python dependencies.
├── data/                 # Directory to store your data files.
│   └── ...
└── README.md             # This file.
``` 