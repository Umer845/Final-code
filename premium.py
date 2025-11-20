import streamlit as st
import pandas as pd
import datetime
from sqlalchemy import create_engine, text
import requests

# ✅ Database configuration
DB_USER = "postgres"
DB_PASSWORD = "United2025"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "AutoMotor_Insurance"
DB_TABLE_NAME = "motor_insurance_data"
DB_TABLE_NAME1 = "premium_results"

# -------------------- Local LLM (Ollama) --------------------
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
            timeout=200
        )
        output_text = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response"' in data:
                    part = data.split('"response":"')[-1].split('"')[0]
                    output_text += part
        return output_text.strip() or "No explanation returned."
    except Exception as e:
        return f"⚠️ Could not connect to local LLM: {e}"

# -------------------- Database Connection --------------------
@st.cache_resource
def get_engine():
    try:
        engine_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        return create_engine(engine_string)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# -------------------- Main App --------------------
def show():
    st.title("🚗 Motor Insurance Premium Prediction")

    engine = get_engine()
    if not engine:
        st.stop()

    # Load model from session state
    model = st.session_state.get("loaded_model", None)
    selected_model = st.session_state.get("selected_model_name", "Unknown Model")
    feature_cols = st.session_state.get("model_features", None)

    if model is None:
        st.warning("⚠️ No model loaded. Please go to 'Model Validation' first.")
        return

    current_year = datetime.datetime.now().year

    # ---------------------------------------------------
    # ✅ FETCH DISTINCT VEHICLE MAKES FROM DATABASE
    # ---------------------------------------------------
    try:
        with engine.connect() as conn:
            df_makes = pd.read_sql(
                text(f"SELECT DISTINCT vehicle_make FROM {DB_TABLE_NAME} ORDER BY vehicle_make ASC"),
                conn
            )
        make_list = df_makes["vehicle_make"].dropna().tolist()
    except Exception as e:
        st.error(f"❌ Failed loading vehicle makes: {e}")
        make_list = []

    st.subheader("Enter Vehicle Details")

    with st.form(key="predict_form"):

        # ---------------------------------------------------
        # ✅ VEHICLE MAKE DROPDOWN
        # ---------------------------------------------------
        vehicle_make = st.selectbox(
            "Vehicle Make",
            options=make_list,
            index=0 if "Toyota" not in make_list else make_list.index("Toyota")
        )

        # ---------------------------------------------------
        # ✅ VEHICLE MODELS LOADED DEPENDING ON MAKE
        # ---------------------------------------------------
        model_list = []
        try:
            with engine.connect() as conn:
                df_models = pd.read_sql(
                    text(f"""
                        SELECT DISTINCT vehicle_model 
                        FROM {DB_TABLE_NAME} 
                        WHERE LOWER(vehicle_make) = LOWER(:make)
                        ORDER BY vehicle_model ASC;
                    """),
                    conn,
                    params={"make": vehicle_make}
                )
            model_list = df_models["vehicle_model"].dropna().tolist()
        except Exception as e:
            st.error(f"❌ Failed loading vehicle models: {e}")

        vehicle_model = st.selectbox(
            "Vehicle Model",
            options=model_list if model_list else ["Corolla"]
        )

        # ---------------------------------------------------
        # 🚫 NOTHING ELSE CHANGED BELOW THIS LINE
        # ---------------------------------------------------
        vehicle_make_year = st.number_input("Vehicle Make Year", 1980, current_year, 2020)
        sum_insured = st.number_input("Sum Insured", 500000)
        risk_profile = st.selectbox(
            "Select Risk Profile",
            ["Low", "Low to Moderate", "Moderate to High", "High"]
        )
        submit = st.form_submit_button("Predict Premium")

    if not submit:
        return

    make_normalized = vehicle_make.strip()
    model_normalized = vehicle_model.strip()
    vehicle_age = current_year - vehicle_make_year

    # --- Fetch Historical Rates ---
    hist_min = hist_max = hist_avg = None
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT MIN(rate) as min_rate, 
                       MAX(rate) as max_rate, 
                       AVG(rate) as avg_rate
                FROM {DB_TABLE_NAME}
                WHERE LOWER(vehicle_make) = LOWER(:make)
                  AND LOWER(vehicle_model) = LOWER(:model);
            """)
            result = pd.read_sql(query, conn, params={"make": make_normalized, "model": model_normalized})
            if not result.empty:
                hist_min = result.iloc[0]["min_rate"]
                hist_max = result.iloc[0]["max_rate"]
                hist_avg = result.iloc[0]["avg_rate"]
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")

    # --------------------------
    # Build inference row
    # --------------------------
    def build_feature_row():
        if isinstance(feature_cols, list) and len(feature_cols) > 0:
            row = pd.DataFrame([[0] * len(feature_cols)], columns=[str(c) for c in feature_cols])

            if "sum_insured" in row.columns:
                row.at[0, "sum_insured"] = float(sum_insured)
            if "vehicle_make_year" in row.columns:
                row.at[0, "vehicle_make_year"] = int(vehicle_make_year)
            if "vehicle_age" in row.columns:
                row.at[0, "vehicle_age"] = int(vehicle_age)

            vmk = f"vehicle_make_{make_normalized}".lower().replace(" ", "_")
            vmd = f"vehicle_model_{model_normalized}".lower().replace(" ", "_")
            for col in [vmk, vmd]:
                if col in row.columns:
                    row.at[0, col] = 1.0

            return row
        else:
            return pd.DataFrame([{
                "VEHICLE MAKE YEAR": int(vehicle_make_year),
                "SUM INSURED": float(sum_insured),
                "vehicle_age": int(vehicle_age)
            }])

    # --------------------------
    # Prediction logic
    # --------------------------
    try:
        X_row = build_feature_row()
        raw_prediction = float(model.predict(X_row)[0])
        predicted_premium = max(raw_prediction, 0.0)
        predicted_rate = (predicted_premium / sum_insured * 100) if sum_insured else 0.0
        source = f"🤖 Based on {selected_model}"
    except Exception as e:
        if hist_avg is not None and sum_insured:
            predicted_rate = float(hist_avg)
            predicted_premium = (predicted_rate / 100.0) * float(sum_insured)
            source = "📊 Based on historical rates (fallback due to model feature mismatch)"
        else:
            st.error(f"❌ Prediction failed and no historical data available: {e}")
            return

    # Age adjustment
    if vehicle_age <= 1:
        age_adj = 0.85
    elif 2 <= vehicle_age <= 5:
        age_adj = 1.00
    elif 6 <= vehicle_age <= 10:
        age_adj = 1.10
    else:
        age_adj = 1.25

    predicted_premium *= age_adj
    predicted_rate = (predicted_premium / sum_insured * 100) if sum_insured else 0.0

    risk_map = {"Low": 0.05, "Low to Moderate": 0.075, "Moderate to High": 0.10, "High": 0.15}
    final_premium = predicted_premium * (1 + risk_map.get(risk_profile, 0))
    final_rate = (final_premium / sum_insured * 100) if sum_insured else 0.0

    # Result Blocks
    st.markdown("### Prediction Results:")

    if hist_min is not None and hist_max is not None:
        st.markdown(
            f"""<div style="background-color:#0d5fafd9; padding:12px; border-radius:8px;">
                <span style="font-weight:bold; color:white;">
                    Minimum Premium Rate: {hist_min:.2f}% || Maximum Premium Rate: {hist_max:.2f}%
                </span>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown(
        f"""<div style="background-color:#1b6706e6; padding:12px; border-radius:8px; margin-top:10px;">
            <span style="font-weight:bold; color:white;">
                Predicted Premium: {predicted_premium:,.2f} || Rate: {predicted_rate:.2f}%
            </span>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""<div style="background-color:#735109; padding:12px; border-radius:8px; margin-top:10px;">
            <span style="font-weight:bold; color:white;">
                Final Premium with Risk: {final_premium:,.2f} || Rate: {final_rate:.2f}%<br/>
                <small>{source}</small>
            </span>
        </div>""",
        unsafe_allow_html=True
    )

    # LLM Explanation
    st.subheader("🧠 Local AI Explanation")
    prompt = f"""
    Explain why a motor insurance case with Risk Profile '{risk_profile}' 
    resulted in a Final Premium of {final_premium:,.2f} ({final_rate:.2f}% of Sum Insured).
    Vehicle Age: {vehicle_age} years, Sum Insured: {sum_insured}.
    """
    explanation = local_llm_explanation(prompt)
    st.write("📖 " + explanation)

    # Save to DB
    try:
        with engine.begin() as conn:
            insert_query = text(f"""
                INSERT INTO {DB_TABLE_NAME1}
                (vehicle_make, vehicle_model, vehicle_make_year, sum_insured, vehicle_age, risk_profile,
                 historical_min_rate, historical_max_rate, historical_avg_rate,
                 predicted_premium, predicted_rate, final_premium, final_rate,
                 prediction_source, created_at, llm_prompt)
                VALUES (:make, :model, :year, :sum, :age, :risk,
                        :hist_min, :hist_max, :hist_avg,
                        :pred_premium, :pred_rate, :final_premium, :final_rate,
                        :source, :created_at, :llm_prompt)
            """)
            conn.execute(insert_query, {
                "make": make_normalized,
                "model": model_normalized,
                "year": int(vehicle_make_year),
                "sum": float(sum_insured),
                "age": int(vehicle_age),
                "risk": risk_profile,
                "hist_min": float(hist_min) if hist_min is not None else None,
                "hist_max": float(hist_max) if hist_max is not None else None,
                "hist_avg": float(hist_avg) if hist_avg is not None else None,
                "pred_premium": float(predicted_premium),
                "pred_rate": float(predicted_rate),
                "final_premium": float(final_premium),
                "final_rate": float(final_rate),
                "source": source,
                "created_at": datetime.datetime.now(),
                "llm_prompt": explanation
            })
        st.success("✅ Prediction saved successfully!")
    except Exception as e:
        st.error(f"❌ Failed to save to DB: {e}")
