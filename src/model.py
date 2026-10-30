import os, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from src.data import load_orders, make_features, FEATURES, CAT_FEATURES, NUM_FEATURES

MODEL_PATH = "artifacts/rto_model.joblib"

def build_model():
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_FEATURES),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), CAT_FEATURES)
    ])
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", prep), ("classifier", clf)])

def train_and_save(data_path="data/orders_sample.csv"):
    df = load_orders(data_path)
    x = make_features(df)[FEATURES]
    y = (df["status_norm"] == "RTO").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(x_train, y_train)
    p = model.predict_proba(x_test)[:, 1]
    yhat = (p >= 0.5).astype(int)
    metrics = {
        "test_rows": int(len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, p)),
        "accuracy": float(accuracy_score(y_test, yhat)),
        "precision_rto": float(precision_score(y_test, yhat, zero_division=0)),
        "recall_rto": float(recall_score(y_test, yhat, zero_division=0)),
        "f1_rto": float(f1_score(y_test, yhat, zero_division=0))
    }
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model, metrics, df

def get_model():
    if not os.path.exists(MODEL_PATH):
        return train_and_save()[0]
    return joblib.load(MODEL_PATH)
