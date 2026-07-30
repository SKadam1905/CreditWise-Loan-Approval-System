import streamlit as st
from utils.ui_components import init_page, render_metric_card
from utils.prediction import load_metadata

# 1. Initialize Page (config, CSS, and sidebar)
init_page("Home")

# 2. Main Content Header
st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem; color: #1e3a8a;'>💳 CreditWise</h1>
        <h3 style='font-size: 1.25rem; font-weight: 400; color: #475569; margin: 0;'>AI-Powered Loan Eligibility Assessment</h3>
    </div>
""", unsafe_allow_html=True)

# 3. Brief Explanation
st.write(
    "Welcome to CreditWise, a professional decision-support tool designed for fintech loan underwriting. "
    "Using an end-to-end Machine Learning pipeline trained on historical applicant profiles, "
    "CreditWise evaluates a borrower's eligibility, predicts the likelihood of loan approval, and highlights core risk observations."
)

st.markdown("<br>", unsafe_allow_html=True)

# 4. Feature Cards
st.subheader("Key Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-card" style="height: 100%;">
            <h4 style="margin-top: 0; color: #1d4ed8;">⚡ Instant Prediction</h4>
            <p style="font-size: 0.9rem; color: #64748b; margin-bottom: 0;">
                Input applicant metrics (income, credit score, DTI, savings) to receive an immediate underwriting prediction.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card" style="height: 100%;">
            <h4 style="margin-top: 0; color: #1d4ed8;">📊 Probability Estimation</h4>
            <p style="font-size: 0.9rem; color: #64748b; margin-bottom: 0;">
                Review fine-grained confidence percentages for approval and rejection, backed by Gradient Boosting probability scores.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card" style="height: 100%;">
            <h4 style="margin-top: 0; color: #1d4ed8;">💡 Risk Observations</h4>
            <p style="font-size: 0.9rem; color: #64748b; margin-bottom: 0;">
                Explore instant rule-based evaluations of key underwriting metrics such as DTI, credit history, and savings coverage.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Project Workflow
st.subheader("Underwriting Workflow")

workflow_cols = st.columns(4)
steps = [
    ("1. Enter Information", "Provide applicant demographics, financial variables, and loan particulars."),
    ("2. Real-time Validation", "Verify that numeric entries, credit scores, and terms match expected scales."),
    ("3. Run Prediction", "Evaluate the profile through the trained and cross-validated ML pipeline."),
    ("4. Review Assessment", "Inspect the recommendation, probability gauges, and explanatory risk flags.")
]

for idx, (step_title, step_desc) in enumerate(steps):
    with workflow_cols[idx]:
        st.markdown(f"""
            <div style="background-color: var(--background-secondary, #f8fafc); border: 1px dashed #cbd5e1; border-radius: 8px; padding: 1rem; height: 100%;">
                <h5 style="margin-top: 0; color: #0f172a;">{step_title}</h5>
                <p style="font-size: 0.85rem; color: #64748b; margin: 0;">{step_desc}</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Model Summary Metrics
st.subheader("Model Validation Summary")
try:
    metadata = load_metadata()
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        render_metric_card("Selected Model", metadata["model_name"], "🤖")
    with m_col2:
        render_metric_card("F1-Score (Test)", f"{metadata['f1_score']*100:.2f}%", "🎯")
    with m_col3:
        render_metric_card("ROC-AUC", f"{metadata['roc_auc']*100:.2f}%", "📈")
    with m_col4:
        render_metric_card("Training Records", f"{metadata['num_training_records']}", "📁")
        
except Exception:
    st.info("Train the model using train_model.py to display advanced validation statistics here.")

st.markdown("<br>", unsafe_allow_html=True)

# 7. Technology Stack & Navigation
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Technology Stack")
    st.markdown("""
        - **Model Pipeline**: Scikit-Learn (ColumnTransformer, Pipeline, Imputers, Standard Scaler, One-Hot Encoder)
        - **Classifier**: Gradient Boosting Classifier
        - **User Interface**: Streamlit (Multi-page configuration, Custom CSS, Plotly/Matplotlib integrations)
        - **Serialization**: Joblib
        - **Data Manipulation**: Pandas, Numpy
    """)

with col_right:
    st.subheader("Get Started")
    st.write("Ready to assess a loan applicant's eligibility?")
    if st.button("Open Applicant Assessment"):
        try:
            st.switch_page("pages/1_Applicant_Assessment.py")
        except Exception:
            # Fallback if switch_page fails or in older streamlit
            st.info("Please click '1 Applicant Assessment' in the sidebar menu on the left.")

# 8. Footer Academic Notice
st.markdown("<hr style='margin-top: 2rem; margin-bottom: 1.5rem; border-color: #e2e8f0;'>", unsafe_allow_html=True)
st.caption(
    "⚠️ **Educational and Portfolio Notice:** "
    "This application is an academic decision-support project and does not provide actual loan approval or financial advice. "
    "Decisions generated here are simulated risk classifications and should not be used as official credit evaluations."
)
