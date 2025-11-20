import streamlit as st  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
import joblib
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sklearn.tree import DecisionTreeRegressor
import requests   # ✅ for local LLM (Ollama)


# -------------------- Page Config --------------------
st.set_page_config(page_title="Motor Insurance Model Training", layout="wide")
st.title("🚗 Motor Insurance Premium Prediction -Training Using AdaBoost Regressor")

# -------------------- DB CONFIG --------------------
DB_USER = "postgres"
DB_PASSWORD = "United2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AutoMotor_Insurance"
DB_TABLE_NAME = "motor_insurance_data"

def get_engine():
    try:
        engine_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(engine_string)
        return engine
    except Exception as e:
        st.error(f"Database connection error: {e}")
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

# -------------------- Local LLM (Ollama) Integration --------------------
def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def local_llm_explanation(prompt, model_name="llama3"):
    """Calls Ollama locally to generate an explanation."""
    if not is_ollama_running():
        return "⚠️ Local LLM not running. Please start Ollama using 'ollama serve'."

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt},
            timeout=180  # increased timeout
        )

        # Ollama streams the response line-by-line
        output_text = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response"' in data:
                    part = data.split('"response":"')[-1].split('"')[0]
                    output_text += part

        return output_text.strip() if output_text else "No explanation returned from local LLM."
    except Exception as e:
        return f"⚠️ Could not connect to local LLM: {e}"

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("📂 Upload your Motor Insurance dataset (CSV)", type=["csv"], key="train_upload")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    engine = get_engine()
    if engine is not None:
        try:
            with engine.begin() as conn:
                df.to_sql(DB_TABLE_NAME, con=conn, if_exists="append", index=False)
            st.success("✅ Dataset saved to PostgreSQL automatically!")
        except SQLAlchemyError as e:
            st.error(f"Database error while saving: {e}")
    else:
        st.error("❌ Could not connect to the database.")

    st.write("### 📑 Dataset Preview")
    st.dataframe(df.head())

    # 🎯 Fixed Target Column
    if "policy_premium" in df.columns:
        target_col = "policy_premium"
        st.success("🎯 Target column automatically set to 'policy_premium'")
    else:
        st.error("❌ 'policy_premium' column not found in dataset.")
        st.stop()

    current_year = 2025
    if "vehicle_make_year" in df.columns:
        df["vehicle_age"] = current_year - df["vehicle_make_year"]
    else:
        st.error("❌ 'vehicle_make_year' column not found in dataset.")
        st.stop()

    if "driver_age" not in df.columns:
        st.warning("⚠ 'driver_age' column missing, filling with default 30.")
        df["driver_age"] = 30

    df["risk_percentage"], df["risk_label"] = zip(*df.apply(lambda r:
        calculate_risk_score(
            r.get("vehicle_use", "personal"),
            r["vehicle_age"],
            r["sum_insured"],
            r["driver_age"]
        ), axis=1))

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------- Train Button --------------------
    if st.button("🚀 Train Model", key="train_btn"):
        with st.spinner("Training AdaBoost model... Please wait ⏳"):
            model = AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=10),  # ✅ Updated keyword
                n_estimators=500,
                learning_rate=0.07,
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            accuracy = 100 * (1 - (abs(y_test - y_pred) / y_test).mean())

        st.subheader("📊 Model Performance")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**MAE:** {mae:,.2f}")
        st.write(f"**RMSE:** {rmse:,.2f}")
        st.write(f"🎯 **Accuracy:** {accuracy:.2f}%")

        # -------------------- Local LLM Explanation --------------------
        st.subheader("🧠 Local AI Explanation of Model Performance")

        prompt = f"""
        You are a data science expert. The AdaBoost regression model trained for motor insurance premium prediction produced:
        - R²: {r2:.4f}
        - MAE: {mae:,.2f}
        - RMSE: {rmse:,.2f}
        - Accuracy: {accuracy:.2f}%

        Explain what these metrics imply about the model performance in predicting premiums,
        and provide short insights on how risk factors might affect prediction accuracy.
        """

        explanation = local_llm_explanation(prompt)
        st.write(explanation)

        # ✅ Save model locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/AdaBoost_premium_model_LLM.pkl")
        joblib.dump(feature_cols, "models/model_features_AdaBoost_LLM.pkl")
        joblib.dump(categorical_cols, "models/model_cat_features_AdaBoost_LLM.pkl")

        # ✅ Store results in PostgreSQL
        try:
            engine = get_engine()
            if engine is not None:
                results_df = pd.DataFrame([{
                    "model_name": "AdaBoostRegressor + Local LLM",
                    "r2_score": round(r2, 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "accuracy": round(accuracy, 4),
                    "prompt": prompt
                }])
                with engine.begin() as conn:
                    results_df.to_sql("model_training_results", con=conn, if_exists="append", index=False)
                st.success("📊 Training results saved to PostgreSQL!")
            else:
                st.warning("⚠ Could not connect to DB for saving results.")
        except SQLAlchemyError as e:
            st.error(f"Database error while saving results: {e}")

        st.success("✅ Model trained and saved in `models/` folder")