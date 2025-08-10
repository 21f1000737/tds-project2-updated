# Data Analyst Agent

A FastAPI-based data analysis agent that uses Claude 3.5 Sonnet to analyze data and answer questions through web scraping and file processing.

## Features

- **URL Scraping**: Scrapes websites and extracts DOM structure and table data
- **File Processing**: Supports CSV, JSON, Parquet, TXT, and image files
- **LLM Integration**: Uses Claude 3.5 Sonnet with tool calling for intelligent analysis
- **Code Execution**: Executes LLM-generated Python code in isolated subprocess
- **JSON Output**: Ensures all responses are in valid JSON format

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export ANTHROPIC_API_KEY=your_claude_api_key_here
   ```

3. **Start the Server**:
   ```bash
   python start.py
   ```
   
   Or directly with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Usage

### API Endpoint

**POST** `/api/`

Upload files including a required `questions.txt` file:

```bash
curl "http://localhost:8000/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  -F "image.png=@image.png"
```

### Health Check

**GET** `/health`

```bash
curl http://localhost:8000/health
```

### Example

Run the example usage script:

```bash
python example_usage.py
```

## File Processing

The API processes different file types:

- **CSV/Parquet**: Extracts columns and 3 sample values per column
- **JSON**: Extracts structure and sample values
- **TXT**: First 1000 lines or full content
- **Images**: Base64 encoded data
- **Other**: Generic text handling

## Function Calling

The LLM has access to:

- `scrape_url(url)`: Scrapes websites and returns DOM structure + tables

## Architecture

1. Upload files are temporarily stored and processed
2. File summaries are generated based on type
3. Gemini 2.0 analyzes the data using function calling
4. LLM generates Python code for analysis
5. Code executes in isolated subprocess
6. Results returned as JSON

## Environment Variables

- `ANTHROPIC_API_KEY`: Required - Your Anthropic Claude API key