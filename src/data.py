import pandas as pd
import numpy as np

TARGET_STATUSES = {"DELIVERED", "RTO"}

def load_orders(path="data/orders_sample.csv"):
    df = pd.read_csv(path)
    df["status_norm"] = df["final_status"].astype(str).str.strip().str.upper()
    # IN_TRANSIT is not a final outcome, so it is excluded from supervised training.
    df = df[df["status_norm"].isin(TARGET_STATUSES)].copy()
    # Keep one record per order_id to avoid counting duplicate IDs as separate outcomes.
    df = df.drop_duplicates(subset=["order_id"], keep="first").copy()
    return df.reset_index(drop=True)

def make_features(df):
    x = df.copy()
    dt = pd.to_datetime(x.get("order_datetime"), errors="coerce")
    x["order_hour"] = dt.dt.hour
    x["order_dayofweek"] = dt.dt.dayofweek
    x["order_month"] = dt.dt.month
    x["pincode"] = x["pincode"].fillna(-1).astype(int).astype(str)
    x["name_len"] = x["customer_name"].fillna("").astype(str).str.len()
    x["address_len"] = x["address"].fillna("").astype(str).str.len()
    x["phone_valid"] = x["phone"].fillna("").astype(str).str.replace(r"\D", "", regex=True).str.len().eq(10).astype(int)
    x["prior_rto_rate"] = (x["customer_prior_rto"] / x["customer_prior_orders"].replace(0, np.nan)).fillna(0)
    x["prior_customer"] = (x["customer_prior_orders"] > 0).astype(int)
    x["address_digits"] = x["address"].fillna("").astype(str).str.count(r"\d")
    x["address_landmark"] = x["address"].fillna("").astype(str).str.contains(
        r"\b(?:near|opp|opposite|mandir|school|bank|chowk|temple|landmark|ke paas|saamne)\b",
        case=False, regex=True).astype(int)
    return x

FEATURES = [
    "order_value", "num_items", "customer_prior_orders", "customer_prior_rto",
    "prior_rto_rate", "prior_customer", "order_hour", "order_dayofweek",
    "order_month", "name_len", "address_len", "phone_valid", "address_digits",
    "address_landmark", "city", "state", "pincode", "utm_source", "courier", "items"
]
CAT_FEATURES = ["city", "state", "pincode", "utm_source", "courier", "items"]
NUM_FEATURES = [c for c in FEATURES if c not in CAT_FEATURES]
