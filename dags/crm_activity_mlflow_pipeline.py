from airflow.decorators import dag, task
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import pandas as pd
import mlflow
import boto3
import os
import logging
import io
from airflow.exceptions import AirflowSkipException
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
RDS_CONNECTION_ID = os.getenv('RDS_CONNECTION_ID')
ACTIVITY_BUCKET_NAME = os.getenv('ACTIVITY_BUCKET_NAME')
ACTIVITY_FILE_KEY = os.getenv('ACTIVITY_FILE_KEY')
PROCESSED_DATA_BUCKET_NAME = os.getenv('PROCESSED_DATA_BUCKET_NAME')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME')
MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME')

@dag(
    dag_id='crm_activity_mlflow_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    default_args={'owner': 'airflow', 'retries': 2},
    tags=['CRM', 'Activity', 'MLflow', 'Machine Learning']
)
def crm_activity_mlflow_pipeline():
    def get_boto3_session():
        return boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )

    @task
    def extract_crm_data() -> pd.DataFrame:
        logging.info("Extracting CRM data from RDS")
        sql_query = "SELECT * FROM crm_data"
        hook = SQLExecuteQueryOperator.get_hook(conn_id=RDS_CONNECTION_ID)
        crm_data = hook.get_pandas_df(sql=sql_query)
        if crm_data.empty:
            logging.warning("CRM data is empty. Skipping downstream tasks.")
            raise AirflowSkipException("No CRM data available")
        return crm_data

    @task
    def extract_activity_data() -> pd.DataFrame:
        logging.info("Extracting activity data from S3")
        session = get_boto3_session()
        s3_client = session.client('s3')
        response = s3_client.list_objects_v2(Bucket=ACTIVITY_BUCKET_NAME, Prefix=ACTIVITY_FILE_KEY)

        if 'Contents' not in response:
            logging.warning("No activity data found in S3. Skipping downstream tasks.")
            raise AirflowSkipException("No activity data in S3")

        all_dfs = []
        for obj in response['Contents']:
            key = obj['Key']
            if not key.endswith('.csv'):
                continue
            file_response = s3_client.get_object(Bucket=ACTIVITY_BUCKET_NAME, Key=key)
            file_content = file_response['Body'].read()
            df = pd.read_csv(io.BytesIO(file_content))
            df['source_file'] = key
            all_dfs.append(df)

        if not all_dfs:
            logging.warning("No valid CSV files found in activity data.")
            raise AirflowSkipException("No valid activity data")

        combined_data = pd.concat(all_dfs, ignore_index=True)
        logging.info(f"Extracted {len(combined_data)} rows from {len(all_dfs)} CSV files")
        return combined_data

    @task
    def merge_and_transform(crm_data: pd.DataFrame, activity_data: pd.DataFrame) -> pd.DataFrame:
        merged_data = pd.merge(crm_data, activity_data, on="customer_id", how="inner")
        transformed_data = merged_data.drop_duplicates()
        return transformed_data

    @task
    def clean_and_feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop_duplicates()
        if 'event_type' in data.columns:
            data = pd.get_dummies(data, columns=["event_type"])
        return data

    @task
    def load_to_s3(processed_data: pd.DataFrame) -> str:
        session = get_boto3_session()
        s3_client = session.client('s3')
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"processed_folder/{timestamp}_processed_data.csv"
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
            processed_data.to_csv(temp_file.name, index=False)
            temp_file.flush()
            s3_client.upload_file(
                Filename=temp_file.name,
                Bucket=PROCESSED_DATA_BUCKET_NAME,
                Key=s3_key
            )
        return s3_key
    
    @task
    def trigger_mlflow_workflow(s3_key: str) -> str:
            logging.info(f"Starting MLflow workflow for data from S3 key: {s3_key}")
            
            session = get_boto3_session()
            s3_client = session.client('s3')
            logging.info(f"Loading data from {PROCESSED_DATA_BUCKET_NAME}/{s3_key}")
            response = s3_client.get_object(Bucket=PROCESSED_DATA_BUCKET_NAME, Key=s3_key)
            data = pd.read_csv(io.BytesIO(response['Body'].read()))
            logging.info(f"Loaded dataset with shape: {data.shape}")

            X = data.drop(columns=["churned"])
            y = data["churned"]
            logging.info(f"Prepared features (X) with shape: {X.shape} and target (y) with shape: {y.shape}")

            logging.info(f"Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            logging.info(f"Set MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

            with mlflow.start_run() as run:
                logging.info(f"Started MLflow run with ID: {run.info.run_id}")
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                logging.info(f"Split data - Training set size: {X_train.shape}, Test set size: {X_test.shape}")

                model = RandomForestClassifier()
                logging.info("Training RandomForestClassifier model...")
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                logging.info(f"Model accuracy: {accuracy:.4f}")

                mlflow.log_metric("accuracy", accuracy)
                mlflow.sklearn.log_model(model, "model")
                logging.info("Logged model and metrics to MLflow")

                return run.info.run_id

    @task
    def register_best_model(run_id: str):
        # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri("http://host.docker.internal:5000")
        client = mlflow.tracking.MlflowClient()
        model_version = client.create_model_version(
            name=MLFLOW_MODEL_NAME,
            source=f"runs:/{run_id}/model",
            run_id=run_id
        )
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=model_version.version,
            stage="Staging"
        )

    # DAG execution flow
    crm = extract_crm_data()
    activity = extract_activity_data()
    merged = merge_and_transform(crm, activity)
    features = clean_and_feature_engineer(merged)
    s3_key = load_to_s3(features)
    mlflow_id = trigger_mlflow_workflow(s3_key)
    register_best_model(mlflow_id)

crm_activity_mlflow_pipeline()
