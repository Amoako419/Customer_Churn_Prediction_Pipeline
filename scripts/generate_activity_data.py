#!/usr/bin/env python3
import argparse
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

EVENT_TYPES = [
    "page_view",
    "click_ad",
    "search",
    "add_to_cart",
    "purchase",
    "logout"
]

def generate_activity_data(n, customer_ids, start_date, end_date):
    fake = Faker()
    records = []
    delta = (end_date - start_date).days

    for i in range(1, n+1):
        cust = random.choice(customer_ids)
        event_time = start_date + timedelta(days=random.randint(0, delta),
                                            seconds=random.randint(0, 86400))
        records.append({
            "event_id": i,
            "customer_id": cust,
            "event_type": random.choice(EVENT_TYPES),
            "event_time": event_time,
            "session_id": fake.uuid4(),
            "page_url": fake.uri_path()
        })

    return pd.DataFrame.from_records(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of events")
    parser.add_argument("--crm-csv", type=str, default="crm_data.csv", help="CRM CSV to pull customer_ids")
    parser.add_argument("--out", type=str, default="activity_data.csv", help="Output CSV path")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    crm_df = pd.read_csv(args.crm_csv)
    customer_ids = crm_df["customer_id"].tolist()

    start = pd.to_datetime(args.start) if args.start else datetime.now() - timedelta(days=90)
    end = pd.to_datetime(args.end) if args.end else datetime.now()

    df = generate_activity_data(args.n, customer_ids, start, end)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} activity records to {args.out}")

if __name__ == "__main__":
    main()
