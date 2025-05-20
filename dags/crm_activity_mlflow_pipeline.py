from airflow.decorators import dag, task
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
# Correct imports for S3 transfer operators
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from airflow.providers.amazon.aws.transfers.local_to_s3 import LocalFilesystemToS3Operator
from datetime import datetime
import pandas as pd
import mlflow

@dag(
    dag_id='crm_activity_mlflow_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    default_args={'owner': 'airflow', 'retries': 2},
    tags=['CRM', 'Activity', 'MLflow', 'Machine Learning'],
    description='Pipeline to extract CRM and activity data, process it, and train/register a machine learning model using MLflow.'
)
def crm_activity_mlflow_pipeline():
    """
    DAG to extract CRM and activity data, process it, and train/register a machine learning model using MLflow.
    """

    @task
    def extract_crm_data() -> pd.DataFrame:
        """
        Extract CRM data from an RDS database.
        """
        # Replace <sql_query> and <connection_id> with appropriate values
        sql_query = "SELECT * FROM crm_data"
        connection_id = "<rds_connection_id>"
        hook = SQLExecuteQueryOperator.get_hook(conn_id=connection_id)
        crm_data = hook.get_pandas_df(sql=sql_query)
        return crm_data

    @task
    def extract_activity_data() -> pd.DataFrame:
        """
        Extract activity data from an S3 bucket.
        """
        # Replace <bucket_name> and <file_key> with appropriate values
        bucket_name = "<activity_bucket_name>"
        file_key = "<activity_file_key>"
        s3_hook = S3Hook(aws_conn_id="<aws_connection_id>")
        activity_data = s3_hook.read_key(key=file_key, bucket_name=bucket_name)
        return pd.read_csv(activity_data)

    @task
    def merge_and_transform(crm_data: pd.DataFrame, activity_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge and transform the CRM and activity data.
        """
        merged_data = pd.merge(crm_data, activity_data, on="user_id", how="inner")
        # Perform transformations (e.g., renaming columns, handling missing values)
        transformed_data = merged_data.fillna(0)
        return transformed_data

    @task
    def clean_and_feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean, deduplicate, and perform feature engineering on the combined data.
        """
        data.drop_duplicates(inplace=True)
        # Example feature engineering: one-hot encoding, scaling, etc.
        data = pd.get_dummies(data, columns=["category"])
        return data

    @task
    def load_to_s3(processed_data: pd.DataFrame):
        """
        Load the processed data into an S3 bucket.
        """
        # Replace <bucket_name> and <file_key> with appropriate values
        bucket_name = "<processed_data_bucket_name>"
        file_key = "processed_data.csv"
        s3_hook = S3Hook(aws_conn_id="<aws_connection_id>")
        s3_hook.load_string(
            string_data=processed_data.to_csv(index=False),
            key=file_key,
            bucket_name=bucket_name,
            replace=True
        )

    @task
    def trigger_mlflow_workflow():
        """
        Trigger an MLflow workflow to train a machine learning model using the data from S3.
        """
        # Replace <bucket_name> and <file_key> with appropriate values
        bucket_name = "<processed_data_bucket_name>"
        file_key = "processed_data.csv"
        s3_hook = S3Hook(aws_conn_id="<aws_connection_id>")
        data = pd.read_csv(s3_hook.read_key(key=file_key, bucket_name=bucket_name))

        # Split data into features and target
        X = data.drop(columns=["target"])
        y = data["target"]

        # Train a model using MLflow
        mlflow.set_tracking_uri("<mlflow_tracking_uri>")
        mlflow.set_experiment("<mlflow_experiment_name>")
        with mlflow.start_run():
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log metrics and artifacts
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")

            return mlflow.active_run().info.run_id

    @task
    def register_best_model(run_id: str):
        """
        Register the best model in the MLflow Model Registry.
        """
        mlflow.set_tracking_uri("<mlflow_tracking_uri>")
        client = mlflow.tracking.MlflowClient()
        model_name = "<mlflow_model_name>"
        model_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id
        )
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

    # Define task dependencies
    crm_data = extract_crm_data()
    activity_data = extract_activity_data()
    merged_data = merge_and_transform(crm_data, activity_data)
    processed_data = clean_and_feature_engineer(merged_data)
    load_to_s3(processed_data)
    mlflow_run_id = trigger_mlflow_workflow()
    register_best_model(mlflow_run_id)


crm_activity_mlflow_pipeline()