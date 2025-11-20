import streamlit as st
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import risk_profile
import premium


# ==============================
# 🌐 Neon Database Configuration
# ==============================
DB_USER = "neondb_owner"
DB_PASSWORD = "npg_RwMkJvDa4x6G"
DB_HOST = "ep-withered-sky-a1cacs7m-pooler.ap-southeast-1.aws.neon.tech"
DB_PORT = "5432"
DB_NAME = "Final code"   # Neon DB name (spaces allowed)

MODEL_DIR = "models"
TRAINING_RESULTS_TABLE = "model_training_results"
RISK_RESULTS_TABLE = "risk_profile_results"
PREMIUM_RESULTS_TABLE = "premium_results"


# ==============================
# 🔗 Database Connection (Neon)
# ==============================
def get_db_connection():
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
        )
        return engine
    except SQLAlchemyError as e:
        st.error(f"❌ Database Connection Failed: {e}")
        return None


# ==============================
# 🔗 MODEL FILE MAPPING
# ==============================
MODEL_FILE_MAP = {
    "GradientBoost Model": "GradientBoost_Model.pkl",
    "LightGBoost Model": "LightGBoost_Model.pkl",
    "XGBoost Model": "XGBoost_Model.pkl",
    "CatBoost Model": "CatBoost_Model.pkl",
    "AdaBoost Model": "AdaBoost_Model.pkl"
}


# ==============================
# 🎯 MAIN SCREEN
# ==============================
def show():

    st.title("📊 Model Validation & Selection")

    engine = get_db_connection()
    if engine is None:
        return

    # --------------------------------------------------------
    # 🚀 LOAD TRAINING RESULTS (ACCURACY TABLE)
    # --------------------------------------------------------
    try:
        df_models = pd.read_sql(
            f"SELECT model_name, accuracy FROM {TRAINING_RESULTS_TABLE};",
            engine
        )

        if df_models.empty:
            st.warning("⚠️ No trained models found in Neon database.")
            return

        st.subheader("📄 Trained Models Summary")
        st.dataframe(df_models, use_container_width=True)

    except SQLAlchemyError as e:
        st.error(f"❌ Could not fetch training results: {e}")
        return
    finally:
        engine.dispose()

    # --------------------------------------------------------
    # 🎛 MODEL SELECTION (Manual Only)
    # --------------------------------------------------------
    model_list = df_models["model_name"].tolist()

    selected_model = st.selectbox(
        "Select a Model to Load",
        options=model_list
    )

    # Show accuracy
    model_accuracy_map = dict(zip(df_models["model_name"], df_models["accuracy"]))
    accuracy_value = model_accuracy_map[selected_model]
    st.metric("Validation Accuracy", f"{accuracy_value:.2f}%")

    # Fix naming pattern
    normalized_model = selected_model.replace("LightGBM", "LightGBoost")

    if normalized_model not in MODEL_FILE_MAP:
        st.error(f"❌ Model not mapped: {normalized_model}")
        return

    # --------------------------------------------------------
    # 📦 LOAD MODEL FILE
    # --------------------------------------------------------
    model_path = os.path.join(MODEL_DIR, MODEL_FILE_MAP[normalized_model])

    if not os.path.exists(model_path):
        st.error(f"🚫 Model file missing: {model_path}")
        return

    loaded_model = joblib.load(model_path)
    st.session_state["loaded_model"] = loaded_model
    st.session_state["selected_model_name"] = normalized_model

    st.success(f"📌 Loaded Model: {os.path.basename(model_path)}")

    # --------------------------------------------------------
    # 📦 LOAD FEATURE COLUMNS
    # --------------------------------------------------------
    features_path = model_path.replace(".pkl", "_features.pkl")

    if os.path.exists(features_path):
        st.session_state["model_features"] = joblib.load(features_path)
        st.info("✅ Feature Set Loaded Successfully")
    else:
        st.session_state["model_features"] = None
        st.warning("⚠ Feature file missing — predictions may vary")

    # --------------------------------------------------------
    # 🔘 ACTION BUTTONS
    # --------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔮 Predict Premium"):
            st.session_state["show_prediction"] = True
            st.session_state["show_results_screen"] = False

    with col2:
        if st.button("📑 View All Stored Results"):
            st.session_state["show_results_screen"] = True
            st.session_state["show_prediction"] = False

    # --------------------------------------------------------
    # 📑 COMBINED RESULTS SCREEN
    # --------------------------------------------------------
    if st.session_state.get("show_results_screen", False):

        st.markdown("---")
        st.subheader("📑 All Stored Prediction Results")

        try:
            engine = get_db_connection()

            df_risk = pd.read_sql(f"SELECT * FROM {RISK_RESULTS_TABLE}", engine)
            df_premium = pd.read_sql(f"SELECT * FROM {PREMIUM_RESULTS_TABLE}", engine)

            if df_risk.empty and df_premium.empty:
                st.info("ℹ️ No results found in Neon database.")
            else:
                df_combined = pd.concat([df_risk, df_premium], axis=1)
                st.dataframe(df_combined, use_container_width=True)

        except SQLAlchemyError as e:
            st.error(f"❌ Could not load results: {e}")
        finally:
            engine.dispose()

    # --------------------------------------------------------
    # 🚦 RUN RISK + PREMIUM MODULES
    # --------------------------------------------------------
    if st.session_state.get("show_prediction", False):

        st.markdown("---")
        st.subheader("🚦 Step 1: Risk Profile Prediction")
        risk_profile.show()

        st.markdown("---")
        st.subheader("💰 Step 2: Premium Prediction")
        premium.show()
