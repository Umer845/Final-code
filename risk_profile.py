import streamlit as st
import psycopg2
from train_model_XGBRegressor_LLM import calculate_risk_score  
from datetime import datetime
import requests   # For local LLM (Ollama)

# -----------------------------------------
# 🌐 Neon PostgreSQL Configuration
# -----------------------------------------
DB_CONFIG = {
    "dbname": "Final code",   # Neon DB name (spaces allowed)
    "user": "neondb_owner",
    "password": "npg_RwMkJvDa4x6G",
    "host": "ep-withered-sky-a1cacs7m-pooler.ap-southeast-1.aws.neon.tech",
    "port": "5432",
    "sslmode": "require"      # IMPORTANT for Neon
}

# -----------------------------------------
# 🎨 Streamlit Styling
# -----------------------------------------
st.markdown("""
<style>
..st-emotion-cache-467cry h4 {
    padding: 0rem 0px 0rem !important;
}
.st-emotion-cache-13na8ym {
    margin-bottom: 0px;
    margin-top: 9px !important;
    width: 100%;
    border-style: solid;
    border-width: 1px;
    border-color: rgba(250, 250, 250, 0.2);
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# 🤖 Local LLM (Ollama) Integration
# -----------------------------------------
def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def local_llm_prompt(prompt, model_name="llama3"):
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

        return output_text.strip() if output_text else "No explanation returned."
    except Exception as e:
        return f"⚠️ Could not connect to local LLM: {e}"


# -----------------------------------------
# 💾 Insert Data into Neon PostgreSQL
# -----------------------------------------
def insert_risk_result(vehicle_use, vehicle_make_year, sum_insured, driver_age,
                       vehicle_age, risk_score, risk_label, llm_prompt):

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        insert_query = """
        INSERT INTO risk_profile_results
        (vehicle_use, vehicle_make_year, sum_insured, driver_age, vehicle_age, 
         risk_score, risk_label, llm_prompt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        cur.execute(insert_query, (
            vehicle_use,
            vehicle_make_year,
            float(sum_insured),
            int(driver_age),
            int(vehicle_age),
            float(risk_score),
            risk_label,
            llm_prompt
        ))

        conn.commit()
        cur.close()
        conn.close()
        return True, None

    except Exception as e:
        return False, str(e)


# -----------------------------------------
# 🚘 Streamlit App UI
# -----------------------------------------
def show():
    st.title("🚗 Motor Insurance Risk Profile")

    with st.form(key="risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            vehicle_use = st.selectbox("Vehicle Use", ["personal", "commercial", "other"])
            vehicle_make_year = st.number_input("Vehicle Make Year", 1980, 2025, 2020)

        with col2:
            sum_insured = st.number_input("Sum Insured", min_value=10000, value=500000)
            driver_age = st.number_input("Driver Age", 16, 100, 30)

        submit = st.form_submit_button("Calculate Risk")

    # -----------------------------------------
    # 🧮 Risk Calculation
    # -----------------------------------------
    if submit:
        current_year = datetime.now().year
        vehicle_age = current_year - vehicle_make_year

        risk_score, risk_label = calculate_risk_score(
            vehicle_use, vehicle_age, sum_insured, driver_age
        )

        # Create LLM explanation
        prompt = (
            f"Explain why this case received a '{risk_label}' risk label "
            f"with score {risk_score:.2f}. "
            f"Vehicle Use: {vehicle_use}, Vehicle Age: {vehicle_age}, "
            f"Driver Age: {driver_age}, Sum Insured: {sum_insured}."
        )
        explanation = local_llm_prompt(prompt)

        clean_explanation = (
            explanation.replace("\n\n", " ")
                       .replace("\n", " ")
                       .replace("\t", " ")
                       .replace("*", " ")
        )

        # Save to Neon DB
        success, error = insert_risk_result(
            vehicle_use, vehicle_make_year, sum_insured, driver_age,
            vehicle_age, risk_score, risk_label, clean_explanation
        )

        if success:
            st.success("✅ Risk profile saved to Neon database.")
        else:
            st.error(f"❌ Database insert error: {error}")

        # -----------------------------------------
        # 🎨 Display Result Boxes
        # -----------------------------------------
        label_colors = {
            "Low": "#4CAF50",
            "Low to Moderate": "#9C27B0",
            "Medium to High": "#FF9800",
            "High": "#F44336"
        }

        # Risk Score
        st.markdown(
            f"""
            <div style="background-color:#2196F3; padding:15px; border-radius:8px;">
                <h4 style="color:white; margin:0;">Risk Score: {risk_score:.2f}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Risk Label
        bg = label_colors.get(risk_label, "#555")
        st.markdown(
            f"""
            <div style="background-color:{bg}; padding:15px; border-radius:8px;">
                <h4 style="color:white; margin:0;">Risk Label: {risk_label}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        # LLM Explanation
        st.subheader("🧠 Local AI Explanation")
        st.write(clean_explanation)


# -----------------------------------------
# ▶️ Run App
# -----------------------------------------
if __name__ == "__main__":
    show()
