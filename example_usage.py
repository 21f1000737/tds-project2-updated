#!/usr/bin/env python3
"""
Example usage script for the Data Analyst Agent API
"""

import requests
import tempfile
import os

def create_sample_questions():
    """Create a sample questions.txt file"""
    questions = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions)
        return f.name

def test_api(api_url="http://localhost:8000/api/"):
    """Test the Data Analyst Agent API"""
    
    # Create sample questions file
    questions_file = create_sample_questions()
    
    try:
        # Prepare files for upload
        files = {
            'files': ('questions.txt', open(questions_file, 'rb'), 'text/plain')
        }
        
        print(f"Testing API at {api_url}")
        print("Sending request with questions.txt...")
        
        # Make the request
        response = requests.post(api_url, files=files, timeout=300)
        
        if response.status_code == 200:
            print("Success! Response received:")
            print(response.json())
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    finally:
        # Clean up
        if os.path.exists(questions_file):
            os.unlink(questions_file)

def test_health_check(api_url="http://localhost:8000"):
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            print("Health check passed:", response.json())
        else:
            print(f"Health check failed: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Health check request failed: {e}")

if __name__ == "__main__":
    print("Data Analyst Agent API Test")
    print("=" * 40)
    
    # Test health check first
    test_health_check()
    print()
    
    # Test main API
    test_api()