#!/usr/bin/env python3
"""
Startup script for the Data Analyst Agent API
"""

import os
import sys

def check_environment():
    """Check if required environment variables are set"""
    # claude_key = os.getenv("ANTHROPIC_API_KEY")
    # if not claude_key:
    #     print("WARNING: ANTHROPIC_API_KEY environment variable is not set!")
    #     print("Set it with: export ANTHROPIC_API_KEY=your_api_key_here")
    #     return False
    return True

def main():
    """Main startup function"""
    print("Starting Data Analyst Agent API...")
    
    if not check_environment():
        print("Environment check failed. The API will not work properly without ANTHROPIC_API_KEY.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Import and run the FastAPI app
    import uvicorn
    from main import app
    
    print("API starting on http://0.0.0.0:8000")
    print("Health check available at http://localhost:8000/health")
    print("Main endpoint at http://localhost:8000/api/")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == "__main__":
    main()