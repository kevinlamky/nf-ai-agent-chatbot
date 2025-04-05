# Azure Deployment Guide for Planning Applications Chatbot

This guide outlines the steps to deploy the Planning Applications Chatbot to Azure App Service using containers.

## Prerequisites

- Azure account with active subscription
- Docker installed locally (for building the image)
- GitHub account (for CI/CD deployment)

## Steps for Deployment via Azure Portal

### 1. Create Azure Resources

1. **Create a Resource Group**:

   - Log in to the [Azure Portal](https://portal.azure.com)
   - Click on "Resource groups" in the left sidebar
   - Click "+ Create" to create a new resource group
   - Enter "planning-app-chatbot-rg" as the name
   - Select your preferred region
   - Click "Review + create" and then "Create"

2. **Create Azure Container Registry (ACR)**:
   - In the Azure Portal, search for "Container registries"
   - Click "+ Create"
   - Select your subscription and the "planning-app-chatbot-rg" resource group
   - Enter a unique registry name (e.g., "planningappchatbotacr")
   - Select a location
   - Choose "Basic" for SKU
   - Click "Review + create" and then "Create"
   - Once created, go to your ACR resource
   - Navigate to "Access keys" under "Settings"
   - Enable "Admin user" and note down the username and password

### 2. Build and Push Docker Image

1. **Build the Docker image locally**:

   ```bash
   docker build -t planningappchatbotacr.azurecr.io/planning-app-chatbot:latest .
   ```

2. **Log in to ACR and push the image**:
   ```bash
   docker login planningappchatbotacr.azurecr.io -u <username> -p <password>
   docker push planningappchatbotacr.azurecr.io/planning-app-chatbot:latest
   ```
   Replace `<username>` and `<password>` with the ACR credentials from the previous step.

### 3. Create and Configure App Service

1. **Create an App Service Plan**:

   - In the Azure Portal, search for "App Service plans"
   - Click "+ Create"
   - Select your subscription and the "planning-app-chatbot-rg" resource group
   - Enter "planning-app-plan" as the name
   - Select "Linux" as the operating system
   - Choose your preferred region
   - Select "B1" as the pricing tier (click on "Pricing tier" to change)
   - Click "Review + create" and then "Create"

2. **Create a Web App**:

   - In the Azure Portal, search for "App Services"
   - Click "+ Create" > "Web App"
   - Select your subscription and the "planning-app-chatbot-rg" resource group
   - Enter "planning-app-chatbot" as the name
   - Select "Docker Container" as the publish option
   - Select "Linux" as the operating system
   - Select your region
   - Select the "planning-app-plan" you created earlier
   - Click "Next: Docker"
   - Select "Single Container" as the options
   - Select "Azure Container Registry" as the image source
   - Select your ACR, image, and tag
   - Click "Review + create" and then "Create"

3. **Configure Environment Variables**:
   - Once the Web App is created, go to the resource
   - Navigate to "Configuration" under "Settings"
   - Click "+ New application setting" to add each of these environment variables:
     - AZURE_OPENAI_ENDPOINT
     - AZURE_OPENAI_API_KEY
     - AZURE_OPENAI_CHAT_MODEL
     - AZURE_OPENAI_API_VERSION
     - AZURE_OPENAI_EMBEDDING_MODEL
     - GOOGLE_API_KEY
     - GOOGLE_CSE_ID
   - Enter the appropriate values for each
   - Click "Save" when finished

### 4. Set Up Continuous Deployment (Optional)

1. **Configure GitHub Actions**:

   - In the Azure Portal, go to your Web App
   - Navigate to "Deployment Center" under "Deployment"
   - Select "GitHub Actions" as the source
   - Authenticate with GitHub and select your repository
   - Select the main branch
   - Review the workflow file and click "Save"

2. **Create GitHub repository secrets**:
   - Go to your GitHub repository
   - Navigate to "Settings" > "Secrets and variables" > "Actions"
   - Add the following secrets:
     - `AZURE_CREDENTIALS`: JSON output from creating a service principal
     - `ACR_LOGIN_SERVER`: Your ACR login server (e.g., planningappchatbotacr.azurecr.io)
     - `ACR_USERNAME`: ACR username
     - `ACR_PASSWORD`: ACR password

## Important Notes

- Ensure all Azure OpenAI and other sensitive credentials are securely managed as environment variables.
- The app should be accessible at `https://planning-app-chatbot.azurewebsites.net` after deployment.
- Consider setting up Azure Key Vault for more secure credential management.
- Use Azure Application Insights for monitoring app performance.

## Troubleshooting

- If deployment fails, check:
  - Container startup logs in the Azure Portal (App Service > Logs > Container logs)
  - Application logs (App Service > Logs > Log stream)
  - Check if all required environment variables are correctly set
  - Verify the container image is properly built and accessible in ACR
