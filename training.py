# training_streamlit_merged.py
"""
Unified Streamlit training dashboard
Combines CatBoost, XGBoost, LightGBM, GradientBoosting, AdaBoost training logic with LLM explanations
Saves models & features to local `models/` and training results to Neon PostgreSQL.

Run: streamlit run training_streamlit_merged.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------
# 🌐 Neon Database Configuration
# ---------------------------------------
DB_USER = "neondb_owner"
DB_PASSWORD = "npg_RwMkJvDa4x6G"
DB_HOST = "ep-withered-sky-a1cacs7m-pooler.ap-southeast-1.aws.neon.tech"
DB_PORT = "5432"
DB_NAME = "Final code"

DB_TABLE_NAME = "motor_insurance_data"
DB_RESULTS_TABLE = "model_training_results"


# ---------------------------------------
# 🔗 Neon Engine
# ---------------------------------------
@st.cache_resource
def get_engine():
    try:
        engine_string = (
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
        )
        return create_engine(engine_string)
    except Exception as e:
        st.error(f"❌ DB Connection Failed: {e}")
        return None


# ---------------------------------------
# 🔢 Risk Scoring (unchanged)
# ---------------------------------------
def calculate_risk_score(vehicle_use, vehicle_age, sum_insured, driver_age):
    if str(vehicle_use).lower() == 'personal':
        vehicleuse_score = 0.2
    elif str(vehicle_use).lower() == 'commercial':
        vehicleuse_score = 1.0
    else:
        vehicleuse_score = 0.6

    if vehicle_age <= 2:
        vehicleage_score = 0.4
    elif 2 <= vehicle_age <= 5:
        vehicleage_score = 0.6
    elif 6 <= vehicle_age <= 8:
        vehicleage_score = 0.8
    else:
        vehicleage_score = 1.0

    if sum_insured <= 300000:
        suminsured_score = 0.2
    elif 300001 <= sum_insured <= 750000:
        suminsured_score = 0.4
    elif 750001 <= sum_insured <= 1500000:
        suminsured_score = 0.6
    elif 1500001 <= sum_insured <= 3000000:
        suminsured_score = 0.8
    else:
        suminsured_score = 1.0

    if driver_age < 25:
        driverage_score = 1.0
    elif 25 <= driver_age <= 35:
        driverage_score = 0.6
    elif 36 <= driver_age <= 55:
        driverage_score = 0.4
    else:
        driverage_score = 1.0

    raw_score = vehicleuse_score + vehicleage_score + suminsured_score + driverage_score

    if 1.2 <= raw_score < 1.8:
        label = "Low"
    elif 1.8 <= raw_score < 2.4:
        label = "Low to Moderate"
    elif 2.4 <= raw_score < 3.0:
        label = "Medium to High"
    else:
        label = "High"

    return raw_score, label


# ---------------------------------------
# 🤖 Local LLM (Ollama)
# ---------------------------------------
def is_ollama_running():
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def local_llm_explanation(prompt, model_name="llama3"):
    if not is_ollama_running():
        return "⚠️ Local LLM not running. Please start Ollama using 'ollama serve'."

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt},
            timeout=180
        )

        output_text = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response"' in data:
                    output_text += data.split('"response":"')[-1].split('"')[0]

        if not output_text:
            try:
                output_text = response.json().get('response', '') or response.text
            except:
                output_text = response.text

        return output_text.strip()
    except Exception as e:
        return f"⚠️ Could not connect to local LLM: {e}"


# ---------------------------------------
# 🔧 Helpers
# ---------------------------------------
def model_already_trained(engine, model_name):
    try:
        query = text(f"""
            SELECT *
            FROM {DB_RESULTS_TABLE}
            WHERE model_name = :m
            ORDER BY id DESC
            LIMIT 1
        """)
        df = pd.read_sql(query, con=engine, params={"m": model_name})
        return df.iloc[0].to_dict() if not df.empty else None
    except:
        return None


def save_features_and_model(model_obj, X_columns, model_name):
    os.makedirs("models", exist_ok=True)
    model_file = f"models/{model_name.replace(' ','_')}.pkl"
    feat_file = f"models/{model_name.replace(' ','_')}_features.pkl"
    joblib.dump(model_obj, model_file)
    joblib.dump(X_columns, feat_file)
    return model_file, feat_file


# ---------------------------------------
# 🏋️ Generic Training Function
# ---------------------------------------
def train_model_generic(df, model_name, model_obj):

    df.columns = df.columns.map(str)

    current_year = datetime.datetime.now().year
    df["vehicle_age"] = current_year - df.get("vehicle_make_year", current_year - 5)
    df["driver_age"] = df.get("driver_age", 30)
    df["risk_percentage"], df["risk_label"] = zip(*df.apply(lambda r:
        calculate_risk_score(
            r.get("vehicle_use", "personal"),
            r["vehicle_age"],
            r["sum_insured"],
            r["driver_age"]
        ), axis=1))

    if "policy_premium" not in df.columns:
        st.error("❌ 'policy_premium' column missing.")
        return

    target = "policy_premium"
    feature_cols = [c for c in df.columns if c != target]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    X.columns = X.columns.map(str)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner(f"Training {model_name}..."):
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    accuracy = float(
        100 * (1 - (np.abs(y_test - y_pred) / y_test)
        .replace([np.inf, -np.inf], np.nan).fillna(0).mean())
    )

    prompt = f"""
    The {model_name} achieved:
    R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%.
    Explain these metrics and their meaning for insurance premium prediction.
    """

    llm_output = local_llm_explanation(prompt)

    save_features_and_model(model_obj, X.columns.tolist(), model_name)

    # ---------------------------------------
    # 💾 Save Results to Neon
    # ---------------------------------------
    try:
        engine = get_engine()

        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_training_results (
                    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    model_name VARCHAR(200),
                    r2_score DOUBLE PRECISION,
                    mae DOUBLE PRECISION,
                    rmse DOUBLE PRECISION,
                    accuracy DOUBLE PRECISION,
                    llm_prompt TEXT,
                    llm_explanation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))

        result_row = pd.DataFrame([{
            "model_name": model_name,
            "r2_score": r2,
            "mae": mae,
            "rmse": rmse,
            "accuracy": accuracy,
            "llm_prompt": prompt.strip(),
            "llm_explanation": llm_output.strip()
        }])

        result_row.to_sql(DB_RESULTS_TABLE, con=engine, if_exists="append", index=False)

        st.success("✅ Training Results Saved to Neon Database")

    except Exception as e:
        st.error(f"❌ DB Save Error: {e}")


# ---------------------------------------
# 🎛️ UI
# ---------------------------------------
def show():
    st.set_page_config(page_title="Unified Training Dashboard", layout="wide")
    st.title("🚀 Unified Model Training — Insurance Premiums")

    engine = get_engine()

    # Check dataset existence
    dataset_exists = False
    try:
        with engine.begin() as conn:
            res = conn.execute(text(f"SELECT COUNT(*) FROM {DB_TABLE_NAME}"))
            dataset_exists = res.scalar() > 0
    except:
        pass

    # Upload dataset
    if not dataset_exists:
        uploaded_file = st.file_uploader("📤 Upload dataset (CSV)", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                df.columns = df.columns.map(str)
                st.dataframe(df.head())

                with engine.begin() as conn:
                    df.to_sql(DB_TABLE_NAME, con=conn, if_exists="append", index=False)

                st.success("✅ Dataset saved to Neon database.")
                st.rerun()
            except Exception as e:
                st.error(f"Upload error: {e}")
            return

    # Dataset exists — load it
    if dataset_exists:
        try:
            df = pd.read_sql_table(DB_TABLE_NAME, con=engine)
            df.columns = df.columns.map(str)
            st.dataframe(df.head())

            st.subheader("🚀 Choose a Model to Train")
            cols = st.columns(5)

            MODELS = [
                ("XGBoost Model", XGBRegressor(
                    n_estimators=500, learning_rate=0.01, max_depth=10,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    objective="reg:squarederror"
                )),
                ("LightGBM Model", LGBMRegressor(
                    n_estimators=500, learning_rate=0.01, max_depth=10,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )),
                ("GradientBoost Model", GradientBoostingRegressor(
                    n_estimators=500, learning_rate=0.01, max_depth=10,
                    subsample=0.8, max_features=0.8, random_state=42
                )),
                ("CatBoost Model", CatBoostRegressor(
                    iterations=4000, learning_rate=0.01, depth=10,
                    loss_function="RMSE", eval_metric="RMSE",
                    random_seed=42, verbose=False
                )),
                ("AdaBoost Model", AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(max_depth=10),
                    n_estimators=500, learning_rate=0.05, random_state=42
                )),
            ]

            for i, (model_name, model_obj) in enumerate(MODELS):
                if cols[i].button(model_name):
                    existing = model_already_trained(engine, model_name)

                    if existing:
                        st.write(f"📊 Latest Results for {model_name}")
                        st.json(existing)
                    else:
                        train_model_generic(df.copy(), model_name, model_obj)

        except Exception as e:
            st.error(f"Dataset load failed: {e}")


if __name__ == "__main__":
    show()
