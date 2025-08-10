import os
import json
import base64
import tempfile
import subprocess
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic

from PIL import Image
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Enhanced Data Analyst Agent", description="API for data analysis using LLMs with retry mechanism")

# Configure Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    model = "claude-sonnet-4-20250514"  # Claude 4 Sonnet
else:
    client = None
    model = None

# Create temp directory for file processing
TEMP_DIR = Path(tempfile.mkdtemp())

# Constants
MAX_RETRIES = 3
SAMPLE_SIZE = 3  # Only 3 rows for context efficiency and cost reduction
CODE_TIMEOUT = 180  # 3 minutes

class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass

class CodeExecutionError(Exception):
    """Custom exception for code execution errors"""
    pass

def scrape_url(url: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Scrape URL and extract DOM structure and tables information with retry mechanism
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract DOM structure (simplified)
            dom_structure = {
                "title": soup.title.string if soup.title else None,
                "headings": {
                    "h1": [h.get_text().strip() for h in soup.find_all('h1')],
                    "h2": [h.get_text().strip() for h in soup.find_all('h2')],
                    "h3": [h.get_text().strip() for h in soup.find_all('h3')]
                },
                "paragraphs_count": len(soup.find_all('p')),
                "links_count": len(soup.find_all('a'))
            }
            
            # Extract tables with improved sampling
            tables_list = []
            tables = soup.find_all('table')
            
            for i, table in enumerate(tables):
                table_info = {
                    "table_index": i,
                    "headers": [],
                    "sample_rows": [],
                    "total_rows": len(table.find_all('tr'))
                }
                
                # Extract headers
                header_row = table.find('tr')
                if header_row:
                    headers = header_row.find_all(['th', 'td'])
                    table_info["headers"] = [h.get_text().strip() for h in headers]
                
                # Extract sample rows (only 3 for cost efficiency)
                rows = table.find_all('tr')[1:4]  # Skip header, take only first 3
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_data = [cell.get_text().strip() for cell in cells]
                    if row_data:
                        table_info["sample_rows"].append(row_data)
                
                tables_list.append(table_info)
            
            return {
                "dom_structure": dom_structure,
                "tables_list": tables_list,
                "url": url,
                "status": "success",
                "attempt": attempt + 1
            }
            
        except Exception as e:
            logger.warning(f"Scraping attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt == max_retries - 1:
                return {
                    "error": str(e),
                    "url": url,
                    "status": "error",
                    "attempts": max_retries
                }
            time.sleep(2 ** attempt)  # Exponential backoff

def get_smart_sample(df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> Dict[str, Any]:
    """
    Get minimal sample of DataFrame (only 3 rows) for context efficiency
    """
    total_rows = len(df)
    
    if total_rows <= sample_size:
        return {
            "sample_data": df.to_dict('records'),
            "is_sample": "False",
            "total_rows": total_rows,
            "sample_rows": total_rows
        }
    
    # For large datasets, just take first 3 rows for cost efficiency
    sample_df = df.head(sample_size)
    
    return {
        "sample_data": sample_df.to_dict('records'),
        "is_sample": "True",
        "total_rows": total_rows,
        "sample_rows": sample_size,
        "sampling_strategy": f"first_{sample_size}_rows_only"
    }

def process_file(file_path: Path, filename: str) -> Dict[str, Any]:
    """
    Process uploaded file and generate enhanced summary based on file type
    """
    file_extension = Path(filename).suffix.lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
            sample_info = get_smart_sample(df)
            
            # Enhanced statistics
            numeric_columns = df.select_dtypes(include=[int, float]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            return {
                "type": "csv",
                "filename": filename,
                "columns": df.columns.tolist(),
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "shape": df.shape,
                "sample_info": sample_info,
                # Remove expensive operations for cost efficiency
                # "null_counts": df.isnull().sum().to_dict(),
                # "basic_stats": df.describe().to_dict() if numeric_columns else {}
            }
            
        
        elif file_extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                # Create DataFrame for consistent sampling
                df = pd.DataFrame(data)
                sample_info = get_smart_sample(df)
                columns = list(df.columns) if not df.empty else []
            elif isinstance(data, dict):
                columns = list(data.keys())
                sample_info = {
                    "sample_data": [data],
                    "is_sample": "False",
                    "total_rows": 1,
                    "sample_rows": 1
                }
            else:
                columns = ["value"]
                sample_info = {
                    "sample_data": [{"value": data}],
                    "is_sample": "False",
                    "total_rows": 1,
                    "sample_rows": 1
                }
                
            return {
                "type": "json",
                "filename": filename,
                "columns": columns,
                "sample_info": sample_info,
                "data_structure": type(data).__name__
            }
        
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
            sample_info = get_smart_sample(df)
            
            numeric_columns = df.select_dtypes(include=[int, float]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            return {
                "type": "parquet",
                "filename": filename,
                "columns": df.columns.tolist(),
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "shape": df.shape,
                "sample_info": sample_info,
                # Remove expensive operations for cost efficiency
                # "null_counts": df.isnull().sum().to_dict(),
                # "basic_stats": df.describe().to_dict() if numeric_columns else {}
            }
            
        
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                total_lines = len(lines)
                
                # Smart sampling for text files
                if total_lines > 1000:
                    sample_lines = lines[:300] + lines[total_lines//2-150:total_lines//2+150] + lines[-300:]
                    content = ''.join(sample_lines)
                    is_sample = "True"
                else:
                    content = ''.join(lines)
                    is_sample = "False"
            
            return {
                "type": "txt",
                "filename": filename,
                "content": content,
                "total_lines": total_lines,
                "is_sample": is_sample,
                "char_count": len(content)
            }
        
        elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            # Get image info without loading full image for large files
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                format_info = img.format
            
            # Only encode small images fully, provide metadata for large ones
            file_size = file_path.stat().st_size
            if file_size < 1024 * 1024:  # Less than 1MB
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                    base64_data = base64.b64encode(image_data).decode('utf-8')
            else:
                base64_data = None
            
            return {
                "type": "image",
                "filename": filename,
                "base64": base64_data,
                "size_bytes": file_size,
                "dimensions": {"width": width, "height": height},
                "mode": mode,
                "format": format_info,
                "is_large": file_size >= 1024 * 1024
            }
        
        else:
            # Generic text file handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                total_size = len(content)
                
                # Smart truncation for unknown files
                if total_size > 10000:
                    content = content[:5000] + "\n...[TRUNCATED]...\n" + content[-2000:]
                    is_truncated = "True"
                else:
                    is_truncated = "False"
            
            return {
                "type": "unknown",
                "filename": filename,
                "content": content,
                "total_size": total_size,
                "is_truncated": is_truncated
            }
            
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        return {
            "type": "error",
            "filename": filename,
            "error": str(e)
        }


def execute_generated_code(code: str, context: Dict[str, Any], max_retries: int = MAX_RETRIES) -> Any:
    """
    Execute LLM-generated code with retry mechanism and better error handling
    """
    for attempt in range(max_retries):
        try:
            # Create a temporary Python file with the code
            temp_file = TEMP_DIR / f"generated_code_{os.getpid()}_{attempt}.py"
            
            # Enhanced code template with better error handling
            # Indent the generated code properly
            indented_code = '\n'.join('    ' + line for line in code.split('\n'))
            
            full_code = f"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import base64
import io
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless operation
plt.switch_backend('Agg')

# Context data
context = {json.dumps(context, default=str)}

try:
    # Generated code
{indented_code}
except Exception as e:
    error_result = {{
        "error": str(e),
        "error_type": type(e).__name__,
        "attempt": {attempt + 1}
    }}
    print(json.dumps(error_result))
    sys.exit(1)
"""
            
            with open(temp_file, 'w') as f:
                f.write(full_code)
            
            # Execute the code
            result = subprocess.run(
                ['python', str(temp_file)],
                capture_output=True,
                text=True,
                timeout=CODE_TIMEOUT,
                cwd=str(TEMP_DIR)  # Set working directory
            )
            
            # Clean up
            temp_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                try:
                    # Try to parse the output as JSON
                    output = result.stdout.strip()
                    if output:
                        parsed_result = json.loads(output)
                        logger.info(f"Code execution successful on attempt {attempt + 1}")
                        return parsed_result
                    else:
                        raise CodeExecutionError("No output from code execution")
                except json.JSONDecodeError as e:
                    # If not valid JSON, return the raw output with error info
                    logger.warning(f"JSON decode error on attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        return {
                            "error": f"Invalid JSON output: {str(e)}", 
                            "raw_output": result.stdout.strip(),
                            "attempts": max_retries
                        }
            else:
                error_msg = result.stderr.strip()
                logger.warning(f"Code execution failed on attempt {attempt + 1}: {error_msg}")
                
                if attempt == max_retries - 1:
                    return {
                        "error": error_msg,
                        "code": result.returncode,
                        "attempts": max_retries,
                        "stdout": result.stdout.strip()
                    }
                
                # Wait before retry
                time.sleep(1)
                
        except subprocess.TimeoutExpired:
            logger.error(f"Code execution timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return {"error": "Code execution timeout", "attempts": max_retries}
        except Exception as e:
            logger.error(f"Execution error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": str(e), "attempts": max_retries}

def generate_analysis_with_retry(questions: str, context: Dict[str, Any], max_retries: int = MAX_RETRIES) -> str:
    """
    Generate analysis code with Claude, including retry mechanism for improved results
    """
    for attempt in range(max_retries):
        try:
            system_prompt = f"""You are an expert data analyst. Generate Python code to analyze the provided data and answer the questions.

Available data:
{json.dumps({k: v for k, v in context.items() if k != 'scraped_data'}, indent=2, default=str)[:3000]}

Scraped data summary:
{json.dumps(context.get('scraped_data', {}), indent=2, default=str)[:2000]}

Requirements:
1. Use the context data provided
2. Answer all questions in the specified format
3. For plots, save as base64 encoded data URIs
4. Always output final results using: print(json.dumps(result))
5. Handle missing data gracefully
6. Include error handling in your code
7. For large datasets, work with the provided samples
8. IMPORTANT: When working with context data, note that boolean values are serialized as strings ("True"/"False"). Convert them as needed using: val == "True"

Questions to answer:
{questions}

Generate complete, executable Python code that processes the data and outputs results as JSON.
"""
            
            messages = [
                {
                    "role": "user",
                    "content": f"Generate Python code to analyze the data and answer the questions. This is attempt {attempt + 1} of {max_retries}."
                }
            ]
            
            if attempt > 0:
                messages[0]["content"] += f"\n\nPrevious attempts failed. Please ensure:\n- Code outputs valid JSON\n- All required libraries are imported\n- Error handling is included\n- Results match the requested format exactly"
            
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages
            )
            
            generated_code = response.content[0].text if response.content else ""
            
            # Extract code from markdown if present
            if "```python" in generated_code:
                code_lines = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                code_lines = generated_code.split("```")[1].split("```")[0].strip()
            else:
                code_lines = generated_code.strip()
            
            if code_lines:
                logger.info(f"Generated code on attempt {attempt + 1} ({len(code_lines)} chars)")
                return code_lines
            else:
                logger.warning(f"Empty code generated on attempt {attempt + 1}")
                
        except Exception as e:
            logger.error(f"Code generation error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise DataProcessingError(f"Failed to generate code after {max_retries} attempts: {str(e)}")
            time.sleep(1)
    
    raise DataProcessingError("Failed to generate valid code")

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    Enhanced main API endpoint for data analysis with retry mechanism
    """
    logger.info("=== API CALL STARTED ===")
    
    if not client:
        raise HTTPException(status_code=500, detail="Claude API key not configured")
    
    try:
        # Process uploaded files
        questions_content = None
        file_summaries = []
        
        logger.info(f"Processing {len(files)} uploaded files...")
        
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            # Save file temporarily
            temp_file_path = TEMP_DIR / file.filename
            with open(temp_file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            # Process based on filename
            if file.filename == "questions.txt":
                questions_content = content.decode('utf-8')
                logger.info(f"Questions loaded: {len(questions_content)} characters")
            else:
                summary = process_file(temp_file_path, file.filename)
                file_summaries.append(summary)
                logger.info(f"File processed: {file.filename} -> {summary.get('type', 'unknown')}")
        
        if not questions_content:
            raise HTTPException(status_code=400, detail="questions.txt file is required")
        
        # Prepare context for LLM (minimal for cost efficiency)
        context = {
            "questions": questions_content,
            "files": file_summaries,
            "available_functions": ["scrape_url"],
            "note": "Only 3 sample rows provided for cost efficiency - analyze full dataset in code"
        }
        
        # Handle URL scraping if needed
        scraped_data = {}
        if "http" in questions_content.lower():
            # Extract URLs from questions
            import re
            urls = re.findall(r'https?://[^\s<>"{}|\\^`[\]]+', questions_content)
            
            for url in urls:
                logger.info(f"Scraping URL: {url}")
                scrape_result = scrape_url(url, max_retries=MAX_RETRIES)
                scraped_data[url] = scrape_result
                logger.info(f"Scraping completed for {url}: {scrape_result.get('status', 'unknown')}")
        
        context["scraped_data"] = scraped_data
        
        # Generate and execute analysis code with retry
        logger.info("=== GENERATING ANALYSIS CODE ===")
        
        # Combined retry mechanism for both generation and execution
        output_log = []
        
        for main_attempt in range(MAX_RETRIES):
            try:
                generated_code = generate_analysis_with_retry(questions_content, context, max_retries=1)
                logger.info(f"Generated code on main attempt {main_attempt + 1} ({len(generated_code)} chars)")
                print(f'\n=== GENERATED CODE ATTEMPT {main_attempt + 1} ===')
                print(generated_code[:500] + ('...' if len(generated_code) > 500 else ''))
                print('=== END GENERATED CODE ===\n')
                
                # Log generated code
                output_log.append(f"=== GENERATED CODE ATTEMPT {main_attempt + 1} ===")
                output_log.append(generated_code)
                output_log.append("=== END GENERATED CODE ===\n")
                
                logger.info("=== EXECUTING ANALYSIS CODE ===")
                result = execute_generated_code(generated_code, context, max_retries=1)
                
                # Log execution result
                output_log.append(f"=== EXECUTION RESULT ATTEMPT {main_attempt + 1} ===")
                output_log.append(json.dumps(result, indent=2, default=str))
                output_log.append("=== END EXECUTION RESULT ===\n")
                
                # Check if execution was successful
                if "error" not in result:
                    logger.info(f"Analysis completed successfully on main attempt {main_attempt + 1}")
                    break
                else:
                    logger.warning(f"Execution failed on main attempt {main_attempt + 1}: {result.get('error', 'Unknown error')}")
                    if main_attempt == MAX_RETRIES - 1:
                        logger.error("All retry attempts exhausted")
                        break
                    
            except Exception as e:
                logger.error(f"Main attempt {main_attempt + 1} failed: {str(e)}")
                output_log.append(f"=== EXCEPTION ATTEMPT {main_attempt + 1} ===")
                output_log.append(str(e))
                output_log.append("=== END EXCEPTION ===\n")
                if main_attempt == MAX_RETRIES - 1:
                    raise
        
        # Save all outputs to file
        try:
            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(f"=== API CALL STARTED AT {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
                f.write(f"Questions: {questions_content}\n\n")
                f.write(f"Context files: {[file_info.get('filename', 'unknown') for file_info in file_summaries]}\n\n")
                f.write("\n".join(output_log))
                f.write(f"\n=== API CALL ENDED AT {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        except Exception as log_error:
            logger.error(f"Failed to write output.txt: {log_error}")
        
        # Clean up temporary files
        for file in files:
            temp_file_path = TEMP_DIR / file.filename
            temp_file_path.unlink(missing_ok=True)
        
        logger.info(f"=== ANALYSIS COMPLETED SUCCESSFULLY ===")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"=== ERROR OCCURRED ===")
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy", 
        "model_configured": client is not None,
        "temp_dir": str(TEMP_DIR),
        "max_retries": MAX_RETRIES,
        "sample_size": SAMPLE_SIZE
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "max_retries": MAX_RETRIES,
        "sample_size": SAMPLE_SIZE,
        "code_timeout": CODE_TIMEOUT,
        "model": model
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)