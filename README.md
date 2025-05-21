# Customer Churn Prediction Pipeline

This project implements an end-to-end machine learning pipeline for predicting customer churn using Apache Airflow, MLflow, and AWS services. The pipeline processes customer data and activity logs to build and deploy a machine learning model that predicts customer churn probability.

## Architecture

The pipeline consists of several components:
- Data ingestion from AWS RDS (CRM data) and S3 (activity logs)
- Data preprocessing and feature engineering
- Model training and evaluation using MLflow
- Model versioning and deployment
- Performance monitoring and logging

## Project Contents

Your Astro project contains the following files and folders:

- dags: This folder contains the Python files for your Airflow DAGs. By default, this directory includes one example DAG:
    - `example_astronauts`: This DAG shows a simple ETL pipeline example that queries the list of astronauts currently in space from the Open Notify API and prints a statement for each astronaut. The DAG uses the TaskFlow API to define tasks in Python, and dynamic task mapping to dynamically print a statement for each astronaut. For more on how this DAG works, see our [Getting started tutorial](https://www.astronomer.io/docs/learn/get-started-with-airflow).
- Dockerfile: This file contains a versioned Astro Runtime Docker image that provides a differentiated Airflow experience. If you want to execute other commands or overrides at runtime, specify them here.
- include: This folder contains any additional files that you want to include as part of your project. It is empty by default.
- packages.txt: Install OS-level packages needed for your project by adding them to this file. It is empty by default.
- requirements.txt: Install Python packages needed for your project by adding them to this file. It is empty by default.
- plugins: Add custom or community plugins for your project to this file. It is empty by default.
- airflow_settings.yaml: Use this local-only file to specify Airflow Connections, Variables, and Pools instead of entering them in the Airflow UI as you develop DAGs in this project.

## Prerequisites

- Python 3.12+
- Docker Desktop
- AWS Account with access to:
  - RDS (MySQL database)
  - S3 buckets
- MLflow server
- Astro CLI (Astronomer's Airflow distribution)

## Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Amoako419/Customer_Churn_Prediction_Pipeline.git
cd Customer_Churn_Prediction_Pipeline
```

2. Create and configure environment variables (.env file):
```env
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region

# Database Connection
RDS_CONNECTION_ID=rds_conn
DB_HOST=your_db_host
DB_PORT=3306
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name

# S3 Bucket Information
ACTIVITY_BUCKET_NAME=your_activity_bucket
ACTIVITY_FILE_KEY=activity-data/
PROCESSED_DATA_BUCKET_NAME=your_processed_data_bucket

# MLflow Configuration
MLFLOW_TRACKING_URI=http://host.docker.internal:5000
MLFLOW_EXPERIMENT_NAME=crm_activity_model
MLFLOW_MODEL_NAME=crm_activity_classifier
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

1. Start the MLflow tracking server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

2. Start Airflow using Astro CLI:
```bash
astro dev start
```

This command will spin up the Airflow components:
- Postgres: Metadata Database
- Scheduler: Task monitoring and triggering
- Webserver: UI and API access
- Triggerer: Managing task execution

3. Access the Airflow UI:
- Open http://localhost:8080 in your browser
- Default credentials: airflow/airflow

4. Set up Airflow Connections:
- Navigate to Admin -> Connections
- Add AWS connection
- Add RDS connection
- Verify MLflow connection

5. Trigger the Pipeline:
- Navigate to DAGs view
- Locate 'crm_activity_mlflow_pipeline'
- Click "Trigger DAG"

## Features

1. Data Processing:
- Automated data ingestion from multiple sources
- Robust feature engineering
- Data quality validation
- Efficient data transformation

2. Model Training:
- RandomForestClassifier with optimized parameters
- Cross-validation and performance metrics
- MLflow experiment tracking
- Model versioning and staging

3. Monitoring:
- Comprehensive logging
- Performance metrics tracking
- Error handling and retries
- Pipeline status monitoring

## Troubleshooting

1. MLflow Issues:
- Ensure MLflow server is running
- Check MLFLOW_TRACKING_URI
- Verify Docker network connectivity

2. AWS Connectivity:
- Verify credentials in .env
- Check S3 bucket permissions
- Validate RDS connection

3. Pipeline Errors:
- Check Airflow task logs
- Verify data schemas
- Monitor resource usage

## Contact

For support or contributions:
- GitHub: @Amoako419
- Report issues on the GitHub repository
