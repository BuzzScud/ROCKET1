# Rocket Trading Group - Google Cloud Run Deployment Guide

## 🚀 Overview
This guide explains how to deploy the Rocket Trading Group web application to Google Cloud Run with GitHub integration for continuous deployment.

## 📋 Prerequisites

### 1. Google Cloud Setup
- Google Cloud Platform account
- Project with billing enabled
- Enable required APIs:
  ```bash
  gcloud services enable cloudbuild.googleapis.com
  gcloud services enable run.googleapis.com
  gcloud services enable containerregistry.googleapis.com
  ```

### 2. GitHub Repository
- Push your code to a GitHub repository
- Ensure all files are committed

### 3. Local Development Tools
- Google Cloud SDK installed
- Docker installed (for local testing)

## 🏗️ Deployment Methods

### Method 1: GitHub Integration (Recommended)

#### Step 1: Connect GitHub to Google Cloud Build
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to Cloud Build → Triggers
3. Click "Connect Repository"
4. Select GitHub and authorize
5. Select your repository

#### Step 2: Create Build Trigger
1. Click "Create Trigger"
2. Configure:
   - **Name**: `rocket-trading-app-deploy`
   - **Event**: Push to branch
   - **Branch**: `^main$` (or your main branch)
   - **Configuration**: Cloud Build configuration file
   - **Location**: Repository
   - **File**: `cloudbuild.yaml`

#### Step 3: Deploy
1. Push changes to your main branch
2. Build will automatically trigger
3. Monitor progress in Cloud Build console
4. Once complete, check Cloud Run console for service URL

### Method 2: Manual Deployment

#### Step 1: Build and Deploy
```bash
# Set your project ID
export PROJECT_ID=your-project-id

# Build the container
gcloud builds submit --tag gcr.io/$PROJECT_ID/rocket-trading-app

# Deploy to Cloud Run
gcloud run deploy rocket-trading-app \
  --image gcr.io/$PROJECT_ID/rocket-trading-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1
```

### Method 3: Local Docker Testing

#### Test locally before deployment:
```bash
# Build the image
docker build -t rocket-trading-app .

# Run locally
docker run -p 8080:8080 rocket-trading-app

# Test the application
curl http://localhost:8080/health
```

## 🌐 Configuration

### Environment Variables
The application uses these environment variables:
- `PORT`: Server port (default: 8080)
- `PYTHONUNBUFFERED`: Ensures logs appear in Cloud Run

### Cloud Run Settings
- **Memory**: 1Gi (can be adjusted based on usage)
- **CPU**: 1 (can be increased for higher traffic)
- **Timeout**: 300 seconds
- **Concurrency**: 80 requests per instance
- **Max Instances**: 10 (auto-scaling)
- **Min Instances**: 0 (scales to zero when not used)

## 📱 Application URLs

After deployment, your application will be available at:
- **Main Dashboard**: `https://your-service-url.run.app/`
- **Admin Panel**: `https://your-service-url.run.app/admin`
- **Login**: `https://your-service-url.run.app/login`
- **Signals**: `https://your-service-url.run.app/signals`
- **API Health**: `https://your-service-url.run.app/health`

## 🔧 API Endpoints

The application provides these API endpoints:
- `GET /health` - Health check
- `GET /api/quote/<symbol>` - Get stock quote
- `GET /api/chart/<symbol>` - Get chart data
- `GET /api/market-overview` - Market overview data
- `GET /api/search/<query>` - Search symbols

## 📊 Monitoring

### Cloud Run Logs
```bash
# View logs
gcloud run services logs tail rocket-trading-app --platform managed --region us-central1

# View specific log entries
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rocket-trading-app"
```

### Metrics
Monitor in Google Cloud Console:
- Cloud Run → Services → rocket-trading-app → Metrics
- Request count, latency, error rate
- Memory and CPU usage

## 🔄 Continuous Deployment

The GitHub integration provides automatic deployment:
1. Push to main branch
2. Cloud Build triggers automatically
3. Builds Docker image
4. Deploys to Cloud Run
5. New version goes live

### Build Status
Monitor builds in Cloud Build console:
- View build history
- Check build logs
- Monitor deployment status

## 🛡️ Security

### Authentication
- Current setup allows unauthenticated access
- For production, consider:
  - Cloud Run authentication
  - Custom authentication in the app
  - API key requirements

### CORS
- CORS is enabled for all origins
- Restrict in production for specific domains

## 💰 Cost Optimization

### Cloud Run Pricing
- Pay only for requests and compute time
- Scales to zero when not in use
- First 2 million requests per month are free

### Optimization Tips
- Use min-instances=0 for dev/test
- Increase min-instances for production (faster cold starts)
- Monitor and adjust CPU/memory based on usage

## 🔍 Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check build logs
gcloud builds log [BUILD_ID]

# Common fixes:
# - Ensure all files are committed
# - Check Dockerfile syntax
# - Verify requirements.txt
```

#### Runtime Errors
```bash
# Check Cloud Run logs
gcloud run services logs tail rocket-trading-app --platform managed --region us-central1

# Common fixes:
# - Check port configuration (must be 8080)
# - Verify environment variables
# - Check Python dependencies
```

#### API Issues
- Yahoo Finance API may have rate limits
- Application includes fallback demo data
- Check logs for API errors

## 📚 Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [GitHub Integration Guide](https://cloud.google.com/build/docs/automating-builds/github/build-repos-from-github)

## 🆘 Support

For issues:
1. Check Cloud Run logs
2. Review build logs in Cloud Build
3. Verify all configuration files
4. Test locally with Docker

---

**Note**: This application includes demo data fallbacks for when external APIs are unavailable, ensuring the application remains functional in all environments. 