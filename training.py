# training_streamlit_merged.py
"""
Unified Streamlit training dashboard
Combines CatBoost, XGBoost, LightGBM, GradientBoosting, AdaBoost training logic with LLM explanations
Saves models & features to local `models/` and training results to PostgreSQL.

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
# ⚙️ Database Configuration
# ---------------------------------------
DB_USER = "postgres"
DB_PASSWORD = "United2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AutoMotor_Insurance"
DB_TABLE_NAME = "motor_insurance_data"
DB_RESULTS_TABLE = "model_training_results"


# -------------------- Engine --------------------
@st.cache_resource
def get_engine():
    try:
        return create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
    except Exception as e:
        st.error(f"❌ DB Connection Failed: {e}")
        return None


# -------------------- Risk Scoring --------------------
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


# -------------------- Local LLM --------------------
def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
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
                    part = data.split('"response":"')[-1].split('"')[0]
                    output_text += part
        if not output_text:
            try:
                output_text = response.json().get('response', '') or response.text
            except Exception:
                output_text = response.text
        return output_text.strip() if output_text else "No explanation returned from local LLM."
    except Exception as e:
        return f"⚠️ Could not connect to local LLM: {e}"


# -------------------- Helpers --------------------
def model_already_trained(engine, model_name):
    try:
        query = text(f"""
            SELECT *
            FROM {DB_RESULTS_TABLE}
            WHERE model_name = :m
            ORDER BY id DESC
            LIMIT 1
        """)
        df = pd.read_sql_query(query, con=engine, params={"m": model_name})
        if not df.empty:
            return df.iloc[0].to_dict()
        return None
    except Exception:
        return None


def save_features_and_model(model_obj, X_columns, model_name):
    os.makedirs("models", exist_ok=True)
    model_file = f"models/{model_name.replace(' ','_')}.pkl"
    feat_file = f"models/{model_name.replace(' ','_')}_features.pkl"
    joblib.dump(model_obj, model_file)
    joblib.dump(X_columns, feat_file)
    return model_file, feat_file


# -------------------- Generic Training --------------------
def train_model_generic(df, model_name, model_obj):
    df.columns = df.columns.map(str)  # ✅ Ensure all string column names

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
        st.error("❌ 'policy_premium' column not found in dataset.")
        return

    target_col = "policy_premium"
    feature_cols = [c for c in df.columns if c != target_col]

    X = pd.get_dummies(df[feature_cols], drop_first=True)
    X.columns = X.columns.map(str)  # ✅ Make sure X columns are string
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with st.spinner(f"Training {model_name}... Please wait ⏳"):
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = float(100 * (1 - (np.abs(y_test - y_pred) / y_test)
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0).mean()))

    prompt = f"""
    The {model_name} for motor insurance premium prediction produced:
    - R²: {r2:.4f}
    - MAE: {mae:,.2f}
    - RMSE: {rmse:,.2f}
    - Accuracy: {accuracy:.2f}%
    Explain what these metrics imply and provide insights on how risk factors affect prediction accuracy.
    """

    llm_output = local_llm_explanation(prompt)

    st.markdown(f"<h3 style='text-align:left;'>📊 {model_name} Performance</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{r2:.4f}")
    c2.metric("MAE", f"{mae:,.2f}")
    c3.metric("RMSE", f"{rmse:,.2f}")
    c4.metric("🎯 Accuracy", f"{accuracy:.2f}%")

    st.markdown(f"""
    <div style="background-color:#1f2937;padding:20px;border-radius:10px;margin-top:20px;">
        <h4 style="color:#93c5fd;">🧠 LLM Explanation</h4>
        <pre style="color:#e5e7eb;white-space:pre-wrap;">{llm_output}</pre>
    </div>
    """, unsafe_allow_html=True)

    model_file, feat_file = save_features_and_model(model_obj, X.columns.tolist(), model_name)

    # ✅ Save results to DB
    try:
        engine = get_engine()
        if engine is not None:
            results_df = pd.DataFrame([{
                "model_name": model_name,
                "r2_score": r2,
                "mae": mae,
                "rmse": rmse,
                "accuracy": accuracy,
                "llm_prompt": prompt.strip(),
                "llm_explanation": llm_output.strip()
            }])

            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS model_training_results (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(150),
                        r2_score DOUBLE PRECISION,
                        mae DOUBLE PRECISION,
                        rmse DOUBLE PRECISION,
                        accuracy DOUBLE PRECISION,
                        llm_prompt TEXT,
                        llm_explanation TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))

            results_df.to_sql(DB_RESULTS_TABLE, con=engine, if_exists="append", index=False)
            st.success("✅ Training results & LLM prompt saved to DB!")
    except SQLAlchemyError as e:
        st.error(f"❌ Could not save training results: {e}")


# -------------------- UI --------------------
def show():
    st.set_page_config(page_title="Unified Training Dashboard", layout="wide")
    st.title("🚀 Unified Model Training — Insurance Premiums")

    engine = get_engine()
    dataset_exists = False

    if engine is not None:
        try:
            with engine.begin() as conn:
                res = conn.execute(text(f"SELECT COUNT(*) FROM {DB_TABLE_NAME}"))
                count = res.scalar()
                dataset_exists = bool(count and count > 0)
        except Exception:
            dataset_exists = False

    if dataset_exists:
        st.warning("⚠️ Dataset already exists in the database.")
        if st.button("🗑 Delete Dataset from DB", key="delete_db_data"):
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {DB_TABLE_NAME}"))
            st.success("✅ Dataset deleted successfully! You can upload a new one now.")
            st.rerun()
    else:
        uploaded_file = st.file_uploader("📤 Upload dataset (CSV format)", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df.columns = (
                    df.columns.str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                    .str.replace("-", "_")
                )
                df.columns = df.columns.map(str)  # ✅ ensure all str
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head())

                with engine.begin() as conn:
                    df.to_sql(DB_TABLE_NAME, con=conn, if_exists="append", index=False)
                st.success("✅ Dataset saved to database!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to upload dataset: {e}")
            return

    if dataset_exists:
        try:
            df = pd.read_sql_table(DB_TABLE_NAME, con=engine)
            df.columns = df.columns.map(str)  # ✅ Fix for quoted_name issue
            df.columns = (
                df.columns.str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("-", "_")
            )
            st.success("📦 Loaded dataset from database.")
            st.dataframe(df.head())

            st.write("### 🚀 Choose a Model to Train")

            # 5 columns → 5 buttons (XGB, LGBM, GBM, CatBoost, AdaBoost)
            cols = st.columns(5)

            models = [
                ("XGBoost Model", XGBRegressor(
                    n_estimators=500, learning_rate=0.01, max_depth=10,
                    random_state=42, subsample=0.8, colsample_bytree=0.8,
                    objective="reg:squarederror"
                )),
                ("LightGBoost Model", LGBMRegressor(
                    n_estimators=500, learning_rate=0.01, max_depth=10,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )),
                ("GradientBoost Model", GradientBoostingRegressor(
                    n_estimators=500, learning_rate=0.01, max_depth=10,
                    random_state=42, subsample=0.8, max_features=0.8
                )),
                ("CatBoost Model", CatBoostRegressor(
                    iterations=5000,
                    learning_rate=0.01,
                    depth=10,
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    random_seed=42,
                    verbose=False
                )),
                ("AdaBoost Model", AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(max_depth=10),
                    n_estimators=500,
                    learning_rate=0.05,
                    random_state=42
                ))
            ]

            for i, (mname, mobj) in enumerate(models):
                with cols[i]:
                    if st.button(mname, key=mname):
                        existing = model_already_trained(engine, mname)
                        if existing:
                            st.markdown(f"<h3 style='text-align:left;'>📊 {mname} Results</h3>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="display:flex; flex-direction:column; gap:16px; margin-top:20px;">
                                <div style="background-color:#0ea5a4;padding:14px;border-radius:10px;color:#041014;flex:1;text-align:center;">
                                    <div style="font-size:14px;font-weight:600;">R² Score</div>
                                    <div style="font-size:22px;font-weight:700;">{existing['r2_score']:.4f}</div>
                                </div>
                                <div style="background-color:#0ea5a4;padding:14px;border-radius:10px;color:#041014;flex:1;text-align:center;">
                                    <div style="font-size:14px;font-weight:600">MAE</div>
                                    <div style="font-size:22px;font-weight:700;">{existing['mae']:,.4f}</div>
                                </div>
                                <div style="background-color:#0ea5a4;padding:14px;border-radius:10px;color:#041014;flex:1;text-align:center;">
                                    <div style="font-size:14px;font-weight:600">RMSE</div>
                                    <div style="font-size:22px;font-weight:700;">{existing['rmse']:,.4f}</div>
                                </div>
                                <div style="background-color:#0ea5a4;padding:14px;border-radius:10px;color:#041014;flex:1;text-align:center;">
                                    <div style="font-size:14px;font-weight:600">🎯 Accuracy</div>
                                    <div style="font-size:22px;font-weight:700;">{existing['accuracy']:.2f}%</div>
                                </div>
                            </div>
                            <div style="background-color:#0ea5a4;padding:18px;border-radius:12px;margin-top:20px;color:#041014;">
                                <div style="font-size:15px;font-weight:600">🧠 LLM Explanation</div>
                                <pre style="white-space:pre-wrap;margin-top:12px;">{existing['llm_explanation']}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            train_model_generic(df.copy(), mname, mobj)

        except Exception as e:
            st.error(f"❌ Could not load dataset from DB: {e}")


if __name__ == "__main__":
    show()
