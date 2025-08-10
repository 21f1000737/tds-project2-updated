#!/bin/bash

# Test script for TechCorp Analytics Data Analysis API
echo "=== Testing TechCorp Analytics Data Analysis ==="

# Start the server in background
echo "Starting server on port 8003..."
uv run uvicorn main:app --host 0.0.0.0 --port 8003 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

# Test if server is running
if ! curl -s http://localhost:8003/health > /dev/null; then
    echo "ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo "Server is running. Testing API..."

# Make the API call with both files
echo "Sending test files and questions to API..."
RESPONSE=$(curl -s -X POST "http://localhost:8003/api/" \
    -F "files=@questions.txt" \
    -F "files=@test_data/datasets/samples/test.csv" \
    -F "files=@test_data/datasets/samples/test.json" \
    --max-time 200)

# Check if API call was successful
if [ $? -eq 0 ]; then
    echo "API call successful!"
    echo "Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    echo "ERROR: API call failed"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Clean up
echo ""
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "=== Test completed ==="