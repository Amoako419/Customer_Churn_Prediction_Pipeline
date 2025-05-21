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
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Silence Git warnings in MLflow
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

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
            # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_tracking_uri("http://host.docker.internal:5000")
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            logging.info(f"Set MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

            with mlflow.start_run() as run:
                logging.info(f"Started MLflow run with ID: {run.info.run_id}")
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
                # Feature engineering
                def engineer_features(df):                    # Convert date columns to datetime and extract features
                    date_columns = ['signup_date', 'last_login', 'churn_date', 'event_time']
                    for col in date_columns:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col])
                            # Extract components from datetime
                            df[f'{col}_year'] = df[col].dt.year
                            df[f'{col}_month'] = df[col].dt.month
                            df[f'{col}_day'] = df[col].dt.day
                            # Drop the original datetime column
                            df = df.drop(columns=[col])
                    
                    # Calculate time-based features
                    if all(col in df.columns for col in ['signup_date_year', 'last_login_year']):
                        # Calculate account age in days using the numeric components
                        df['account_age_days'] = (
                            (df['last_login_year'] - df['signup_date_year']) * 365 +
                            (df['last_login_month'] - df['signup_date_month']) * 30 +
                            (df['last_login_day'] - df['signup_date_day'])
                        )

                    
                    # Aggregate event type features
                    event_cols = [col for col in df.columns if col.startswith('event_type_')]
                    df['total_events'] = df[event_cols].sum(axis=1)
                    
                    # Calculate engagement score
                    df['engagement_score'] = (
                        (df['event_type_purchase'] if 'event_type_purchase' in df else 0) * 5 +
                        (df['event_type_add_to_cart'] if 'event_type_add_to_cart' in df else 0) * 3 +
                        (df['event_type_page_view'] if 'event_type_page_view' in df else 0)
                    )
                    
                    return df

                # Apply feature engineering
                X = engineer_features(X)
                  # Drop non-predictive columns
                columns_to_drop = ['name', 'email', 'session_id', 'page_url', 'source_file', 'event_id']
                X = X.drop(columns=[col for col in columns_to_drop if col in X.columns], errors='ignore')

                # Identify feature types
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = X.select_dtypes(include=['object']).columns
                
                # Create preprocessing steps with robust scaling for numeric features
                numeric_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ])
                
                categorical_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ])

                # Combine preprocessing steps
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ],
                    remainder='passthrough'
                )

                # Create pipeline
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier())
                ])

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                logging.info(f"Split data - Training set size: {X_train.shape}, Test set size: {X_test.shape}")

                logging.info("Training model with preprocessing pipeline...")
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:,1]
                
                # Calculate various metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Log all metrics
                logging.info(f"Model Performance Metrics:")
                logging.info(f"Accuracy: {accuracy:.4f}")
                logging.info(f"F1 Score: {f1:.4f}")
                logging.info(f"Recall: {recall:.4f}")
                logging.info(f"Precision: {precision:.4f}")
                logging.info(f"ROC AUC: {roc_auc:.4f}")
                logging.info(f"Confusion Matrix:\n{conf_matrix}")
                
                # Log metrics to MLflow
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("roc_auc", roc_auc)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.sklearn.log_model(model, "model")
                logging.info("Logged model and metrics to MLflow")

                return run.info.run_id

    @task(retries=3, retry_delay=50)
    def register_best_model(mlflow_run_id: str):
        import time
        from mlflow.exceptions import MlflowException
        max_retries = 3
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                mlflow.set_tracking_uri("http://host.docker.internal:5000")
                client = mlflow.tracking.MlflowClient()
                model_version = client.create_model_version(
                    name=MLFLOW_MODEL_NAME,
                    source=f"runs:/{mlflow_run_id}/model",
                    run_id=mlflow_run_id
                )
                client.transition_model_version_stage(
                    name=MLFLOW_MODEL_NAME,
                    version=model_version.version,
                    stage="Staging"
                )
                return
            except MlflowException as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    # DAG execution flow
    crm = extract_crm_data()
    activity = extract_activity_data()
    merged = merge_and_transform(crm, activity)
    features = clean_and_feature_engineer(merged)
    s3_key = load_to_s3(features)
    mlflow_id = trigger_mlflow_workflow(s3_key)
    register_best_model(mlflow_id)

crm_activity_mlflow_pipeline()
