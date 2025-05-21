from airflow.decorators import dag, task
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.http.hooks.http import HttpHook
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Dict
import logging

city_name = 'Munich'
# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv('OPENWEATHER_API_KEY')
# if not api_key:
#     # In a real DAG, you might want to log an error and exit gracefully or let Airflow fail.
#     # Raising an exception here will prevent the DAG from parsing.
#     # Consider using Airflow Variables or Secrets for API keys instead of .env in production.
#     raise ValueError("OPENWEATHER_API_KEY not found in environment variables")

@dag(
    dag_id="fetch_weather_data_with_sensor",
    start_date=datetime(2025, 5, 19),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "goat", "retries": 2},
    tags=["weather", "postgres", "api"],
    description="Fetch weather data from OpenWeather API and insert into PostgreSQL database with API availability check.",
)
def fetch_weather_data_with_sensor():
    """
    DAG to fetch weather data from OpenWeather API and insert it into a PostgreSQL database.
    Includes an HTTP sensor to check API availability before fetching data.
    """

    # Task to check if the OpenWeather API endpoint is available
    check_api_availability = HttpSensor(
        task_id="check_api_availability",
        http_conn_id="openweather_api",  # Connection ID for OpenWeather API
        endpoint=f'2.5/weather?q={city_name}&appid={api_key}',  # Base endpoint to check
        response_check=lambda response: response.status_code == 200,
        poke_interval=30,  # Time interval between checks
        timeout=300,  # Timeout for the sensor
    )

    @task
    def fetch_weather_data(city: str) -> Dict:
        """
        Fetch weather data for a given city from the OpenWeather API.
        """
        http_hook = HttpHook(http_conn_id="openweather_api", method="GET")
        endpoint = f'2.5/weather?q={city}&appid={api_key}' # Use the passed city parameter
        response = http_hook.run(endpoint)
        if response.status_code != 200:
            # Log response content for debugging failed API calls
            logging.error(f"Failed to fetch weather data: {response.text}")
            raise Exception(f"Failed to fetch weather data: Status {response.status_code}")
        return response.json()

    @task
    def process_weather_data(raw_data: Dict) -> Dict:
        """
        Process raw weather data to extract relevant fields for database insertion.
        """
        logging.info(f"Raw weather data: {raw_data}")
        # Add checks for nested keys if they might be missing
        try:
            processed_data = {
                "city": raw_data["name"],
                "temperature": raw_data["main"]["temp"],
                "humidity": raw_data["main"]["humidity"],
                "weather_description": raw_data["weather"][0]["description"],
                "timestamp": raw_data["dt"],
            }
            logging.info(f"Processed weather data: {processed_data}")
            return processed_data
        except KeyError as e:
            logging.error(f"Error processing weather data: Missing key {e}")
            raise

    @task
    def insert_weather_data_to_db(processed_data: Dict):
        """
        Insert processed weather data into the PostgreSQL database using PostgresHook.
        """
        postgres_hook = PostgresHook(postgres_conn_id="postgres_default")  # Connection ID for PostgreSQL
        insert_query = """
            INSERT INTO weather_data (city, temperature, humidity, weather_description, timestamp)
            VALUES (%s, %s, %s, %s, to_timestamp(%s));
        """
        # Ensure data types match your table schema, especially for timestamp
        # OpenWeather's 'dt' is a Unix timestamp (integer)
        # You might need to convert it to a timestamp/datetime object depending on your DB schema
        # For simplicity here, assuming your column accepts integer timestamps or that the hook handles conversion
        # A safer approach might be: datetime.fromtimestamp(processed_data["timestamp"])
        data_to_insert = (
            processed_data["city"],
            processed_data["temperature"],
            processed_data["humidity"],
            processed_data["weather_description"],
            processed_data["timestamp"],
        )
        logging.info(f"Inserting data into database: {data_to_insert}")
        postgres_hook.run(insert_query, parameters=data_to_insert)
        logging.info("Data insertion complete.")


    # Define the task dependencies using TaskFlow API
    # When using @task decorators, calling the function creates the task instance.
    # Dependencies between @task decorated tasks are automatically handled by passing
    # the output of one task as the input to the next.

    # Call the decorated functions to create task instances
    # The variables here represent the *task instances*, not the data results yet.
    api_data_task = fetch_weather_data(city=city_name) # Pass parameter explicitly
    processed_data_task = process_weather_data(api_data_task)
    insert_weather_data_to_db_task = insert_weather_data_to_db(processed_data_task)

    # Define the dependency between the standard sensor task and the first @task task
    # The error occurs because you were chaining the *results* of the @task calls (api_data, processed_data)
    # using >>, which is not how dependencies are typically defined *between* @task tasks.
    # The dependency chain is implicitly built by the data flow in the lines above.
    check_api_availability >> api_data_task


# Instantiate the DAG
fetch_weather_data_with_sensor()