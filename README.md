# TUIK Employment Statistics MCP Server

This MCP (Model Context Protocol) server provides access to Turkish Statistical Institute (TUIK) employment statistics through two main functions that work together to analyze user questions and provide relevant data.

## Features

### 1. Question Analysis and File Selection
- Analyzes user questions about employment statistics
- Automatically selects relevant files from available TUIK datasets
- Uses keyword matching and contextual analysis
- Returns structured JSON with selected files and reasoning

### 2. Data Download and Conversion
- Downloads selected Excel files from TUIK
- Converts Excel data to JSON format
- Provides metadata about the datasets
- Handles multiple file formats (.xlsx, .xls)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the `tuik_scraper` module available in your Python path.

3. Ensure the `istihdam.json` file is present with the list of available files.

## Usage

### Starting the Server

```bash
python test.py
```

The server will start on `http://0.0.0.0:8050`

### Available Tools

#### 1. `analyze_question_and_select_files`

**Purpose**: Analyze user questions and select relevant employment statistics files.

**Input**: 
- `user_question` (string): The user's question about employment statistics

**Output**: JSON string containing:
- `kategori`: The data category
- `selected_files`: List of relevant file names
- `reasoning`: Explanation for file selection
- `total_available`: Total number of available files
- `selected_count`: Number of files selected

**Example**:
```python
question = "Türkiye'de genç işsizlik oranları nasıl?"
result = await analyze_question_and_select_files(question)
```

#### 2. `download_and_convert_files`

**Purpose**: Download selected files and convert them to JSON format.

**Input**: 
- `selected_files_json` (string): JSON output from the first function

**Output**: JSON string containing:
- `download_summary`: Statistics about download success/failure
- `data`: Converted data in JSON format
- `failed_files`: List of files that failed to download
- `data_summary_for_llm`: Sample data for LLM analysis

**Example**:
```python
download_result = await download_and_convert_files(selected_files_json)
```

## Workflow

1. **User asks a question** about employment statistics
2. **Question Analysis**: The first function analyzes the question and selects relevant files
3. **Data Retrieval**: The second function downloads and converts the selected files
4. **LLM Analysis**: The converted data can be sent to an LLM for analysis and answer generation

## Example Questions

The server can handle various types of employment-related questions:

- Youth employment: "Türkiye'de genç işsizlik oranları nasıl?"
- Gender wage gap: "Kadın ve erkek arasındaki ücret farkı nedir?"
- Education and employment: "Eğitim seviyesine göre istihdam durumu nasıl?"
- Regional statistics: "İllere göre işsizlik oranları"
- Sector analysis: "Hangi sektörlerde daha çok istihdam var?"

## File Selection Logic

The system uses keyword matching to select relevant files:

- **Age-related**: Searches for files containing "yaş", "genç" (age, youth)
- **Education-related**: Searches for files containing "eğitim", "okul" (education, school)
- **Wage-related**: Searches for files containing "ücret", "kazanç" (wage, earnings)
- **Unemployment**: Searches for files containing "işsiz" (unemployed)
- **Employment**: Searches for files containing "istihdam", "çalışan" (employment, worker)
- **Gender**: Searches for files containing "cinsiyet" (gender)

## Data Processing

- Excel files (.xlsx, .xls) are converted to JSON format using pandas
- Metadata is extracted including row count, column count, and column names
- Sample data (first 5 rows) is provided for LLM analysis
- Error handling for failed downloads and conversion issues

## Integration with LLMs

The server is designed to work with Language Models:

1. Send user questions to the first function
2. Get selected files and download them with the second function
3. Send the converted data to your preferred LLM (OpenAI, Anthropic, etc.)
4. Get analyzed responses based on the actual TUIK data

## Error Handling

- Graceful handling of download failures
- Excel conversion error management
- Detailed error reporting in JSON responses
- Fallback file selection when no matches are found

## Configuration

- Server runs on port 8050 by default
- Maximum of 5 files selected per query
- UTF-8 encoding for Turkish character support

## Dependencies

- `mcp[cli]>=1.6.0`: Model Context Protocol framework
- `pandas`: Data manipulation and Excel reading
- `openpyxl`: Excel file support (.xlsx)
- `xlrd`: Legacy Excel file support (.xls)
- `httpx`: HTTP client
- `fastmcp`: Fast MCP server implementation

## Files Structure

```
├── test.py              # Main MCP server implementation
├── istihdam.json        # Available files data
├── requirements.txt     # Python dependencies
├── example_usage.py     # Usage examples
└── README.md           # This file
``` 