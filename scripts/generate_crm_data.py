#!/usr/bin/env python3
import argparse
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

def generate_crm_data(n):
    fake = Faker()
    records = []
    for i in range(1, n+1):
        signup_date = fake.date_between(start_date='-2y', end_date='today')
        # churn probability increases with tenure
        churned = fake.boolean(chance_of_getting_true=20 if (datetime.now().date() - signup_date).days < 180 else 40)
        churn_date = fake.date_between(start_date=signup_date, end_date='today') if churned else None

        records.append({
            "customer_id": i,
            "name": fake.name(),
            "email": fake.email(),
            "signup_date": signup_date,
            "plan": random.choice(["free", "basic", "pro", "enterprise"]),
            "monthly_spend": round(random.uniform(0, 500), 2),
            "last_login": fake.date_time_between(start_date=signup_date, end_date='now'),
            "churned": churned,
            "churn_date": churn_date
        })

    return pd.DataFrame.from_records(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of CRM records")
    parser.add_argument("--out", type=str, default="crm_data.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_crm_data(args.n)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} CRM records to {args.out}")

if __name__ == "__main__":
    main()
