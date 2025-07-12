#!/bin/bash

# Rocket Trading Group - Deployment Script
# This script helps deploy the application to Google Cloud Run

set -e

echo "🚀 Rocket Trading Group - Deployment Script"
echo "=============================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    print_error "No Google Cloud project set. Please set it with:"
    echo "gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

print_status "Using project: $PROJECT_ID"

# Check if required APIs are enabled
print_header "Checking required APIs..."
APIS=("cloudbuild.googleapis.com" "run.googleapis.com" "containerregistry.googleapis.com")

for api in "${APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        print_status "$api is enabled"
    else
        print_warning "$api is not enabled. Enabling now..."
        gcloud services enable "$api"
        print_status "Enabled $api"
    fi
done

# Build and deploy
print_header "Building and deploying application..."

# Build the container
print_status "Building container image..."
gcloud builds submit --tag "gcr.io/$PROJECT_ID/rocket-trading-app"

# Deploy to Cloud Run
print_status "Deploying to Cloud Run..."
gcloud run deploy rocket-trading-app \
    --image "gcr.io/$PROJECT_ID/rocket-trading-app" \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --min-instances 0 \
    --timeout 300 \
    --concurrency 80 \
    --set-env-vars PORT=8080

# Get the service URL
SERVICE_URL=$(gcloud run services describe rocket-trading-app --platform managed --region us-central1 --format 'value(status.url)')

print_header "Deployment Complete! 🎉"
echo ""
print_status "Your application is now live at:"
echo "  $SERVICE_URL"
echo ""
print_status "Available endpoints:"
echo "  Main Dashboard: $SERVICE_URL/"
echo "  Admin Panel:    $SERVICE_URL/admin"
echo "  Login:          $SERVICE_URL/login"
echo "  Signals:        $SERVICE_URL/signals"
echo "  API Health:     $SERVICE_URL/health"
echo ""
print_status "API endpoints:"
echo "  GET $SERVICE_URL/api/quote/<symbol>"
echo "  GET $SERVICE_URL/api/chart/<symbol>"
echo "  GET $SERVICE_URL/api/market-overview"
echo "  GET $SERVICE_URL/api/search/<query>"
echo ""
print_status "To view logs:"
echo "  gcloud run services logs tail rocket-trading-app --platform managed --region us-central1"
echo ""
print_status "Deployment completed successfully!" 