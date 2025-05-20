import pandas as pd
import pymysql
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# === CONFIG ===
CSV_FILE = "../data/crm_data/crm_data_3.csv"
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
TABLE_NAME = "crm_data"

# === READ CSV ===
df = pd.read_csv(CSV_FILE)

# Replace NaNs with None to handle nulls
df = df.where(pd.notnull(df), None)

# === CONNECT TO MYSQL RDS ===
engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# === LOAD DATA ===
with engine.begin() as connection:
    df.to_sql(TABLE_NAME, con=connection, if_exists='append', index=False)
    print(f"Inserted {len(df)} records into {TABLE_NAME}.")
