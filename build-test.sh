#!/bin/bash

# Test Docker build locally before deploying to Google Cloud Run
set -e

echo "🧪 Testing Docker Build Locally"
echo "================================="

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile -t rocket-trading-app-test .

echo "✅ Docker build successful!"

# Test the container
echo "Testing container..."
docker run --rm -p 8080:8080 --name rocket-test -d rocket-trading-app-test

# Wait a moment for startup
sleep 3

# Test health endpoint
echo "Testing health endpoint..."
if curl -f http://localhost:8080/health; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    exit 1
fi

# Stop the test container
docker stop rocket-test

echo "✅ All tests passed! Ready for deployment."
echo "Run './deploy.sh' to deploy to Google Cloud Run." 