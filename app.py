import streamlit as st
import dashboard,training, model_Validation

# --- Page Config must be first ---
st.set_page_config(page_title="Insurance Underwriting System", layout="wide")

# --- CSS for styling (keep your design as is) ---
st.markdown("""
<style>
.st-emotion-cache-zuyloh {  
    border: none; 
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    width: 100%;
    height: 100%;
    overflow: visible;
}
.st-emotion-cache-6ms01g {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 400;
    padding: 8px 12px;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 4px 0;
    line-height: 1.6;
    font-size: inherit;
    font-family: inherit;
    color: inherit;
    width: 200px;
    cursor: pointer;
    background-color: rgb(43, 44, 54);
    border: 1px solid rgba(250, 250, 250, 0.2);
}
.st-emotion-cache-6ms01g:hover {
    border-color: #4CAF50;
    color: #4CAF50;
}
.st-emotion-cache-6ms01g:active {
    color: #fff;
    border-color: #4CAF50;
    background-color: #4CAF50;
}
.st-emotion-cache-6ms01g:focus:not(:active) {
    border-color: #4CAF50;
    color: #4CAF50;
}
.st-emotion-cache-9ajs8n h4 {
    font-size: 14px;
    font-weight: 600;
    padding: 0px !important;
}

</style>
""", unsafe_allow_html=True)

# --- Initialize page state ---
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ---- SIDEBAR ----
with st.sidebar:
    # st.image("https://i.postimg.cc/W4ZNtNxP/usti-logo-1.png", use_container_width=True)
    st.markdown("Welcome To LLM Underwritter System")

    st.title("Navigation")

    if st.button("Dashboard"):
      st.session_state.page = "Dashboard"
    if st.button("Training"):
      st.session_state.page = "training"
    if st.button("Model Validation"):
      st.session_state.page = "model_Validation"
    if st.button("Logout"):
      st.session_state.page = "Logout"


# --- Render selected page ---
if st.session_state.page == "Dashboard":
    dashboard.show()
elif st.session_state.page == "training":
    training.show()
elif st.session_state.page == "model_Validation":
    model_Validation.show()
elif st.session_state.page == "Logout":
    st.success("You have been logged out successfully.")
