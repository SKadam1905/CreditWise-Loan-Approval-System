import streamlit as st
import pandas as pd
import io
import csv
import plotly.graph_objects as go
from utils.ui_components import init_page, render_metric_card
from utils.prediction import load_metadata, make_prediction
from utils.validation import validate_inputs

# 1. Initialize Page
init_page("Applicant Assessment")

# 2. Main Headers
st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='font-size: 2.25rem; margin-bottom: 0.5rem; color: #1e3a8a;'>📋 Applicant Assessment</h1>
        <h3 style='font-size: 1.1rem; font-weight: 400; color: #475569; margin: 0;'>fintech loan underwriting and eligibility assessment</h3>
    </div>
""", unsafe_allow_html=True)

# 3. Load metadata and display errors if missing
try:
    metadata = load_metadata()
    categorical_options = metadata["categorical_options"]
    num_ranges = metadata["numerical_ranges"]
except Exception as e:
    st.error("⚠️ Model configuration files not found. Please run the model training script (`train_model.py`) first.")
    st.stop()

# 4. Underwriting Form Layout
st.markdown("### Underwriting Application Form")
st.write("Please complete the following details. Fields are pre-populated with training medians.")

form_placeholder = st.container()

with form_placeholder:
    # We split into three sections using columns/tabs
    personal_col, financial_col, loan_col = st.columns(3)
    
    with personal_col:
        st.subheader("👤 Personal Information")
        
        age = st.slider(
            "Applicant Age",
            min_value=18,
            max_value=100,
            value=int(num_ranges["Age"]["median"]),
            help="Applicant age. The historical dataset covers ages 21 to 59."
        )
        
        gender = st.radio(
            "Gender",
            options=categorical_options["Gender"],
            horizontal=True
        )
        
        marital_status = st.radio(
            "Marital Status",
            options=categorical_options["Marital_Status"],
            horizontal=True
        )
        
        education_level = st.selectbox(
            "Education Level",
            options=categorical_options["Education_Level"]
        )
        
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=15,
            value=int(num_ranges["Dependents"]["median"]),
            step=1,
            help="Number of dependent individuals relying on applicant's income."
        )
        
        employment_status = st.selectbox(
            "Employment Status",
            options=categorical_options["Employment_Status"]
        )
        
        employer_category = st.selectbox(
            "Employer Category",
            options=categorical_options["Employer_Category"]
        )

    with financial_col:
        st.subheader("💵 Financial Information")
        
        applicant_income = st.number_input(
            "Applicant Annual Income ($)",
            min_value=0,
            max_value=1000000,
            value=int(num_ranges["Applicant_Income"]["median"]),
            step=1000,
            help="Annual income of the primary applicant before tax."
        )
        
        coapplicant_income = st.number_input(
            "Coapplicant Annual Income ($)",
            min_value=0,
            max_value=1000000,
            value=int(num_ranges["Coapplicant_Income"]["median"]),
            step=1000,
            help="Annual income of the secondary co-borrower, if applicable."
        )
        
        savings = st.number_input(
            "Applicant Savings ($)",
            min_value=0,
            max_value=5000000,
            value=int(num_ranges["Savings"]["median"]),
            step=1000,
            help="Available liquid assets/savings of the applicant."
        )
        
        credit_score = st.slider(
            "Credit Score (FICO)",
            min_value=300,
            max_value=850,
            value=int(num_ranges["Credit_Score"]["median"]),
            help="Applicant credit score. Valid FICO scores are 300 to 850. Historical dataset contains scores 550 to 799."
        )
        
        existing_loans = st.number_input(
            "Number of Existing Loans",
            min_value=0,
            max_value=20,
            value=int(num_ranges["Existing_Loans"]["median"]),
            step=1,
            help="Number of credit cards, mortgages, or active personal loans."
        )
        
        dti_ratio = st.slider(
            "Debt-to-Income (DTI) Ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(num_ranges["DTI_Ratio"]["median"]),
            step=0.01,
            help="Ratio of applicant's monthly debt payments to gross monthly income. Historical dataset scale is 0.1 to 0.6."
        )
        
        collateral_value = st.number_input(
            "Collateral Value ($)",
            min_value=0,
            max_value=10000000,
            value=int(num_ranges["Collateral_Value"]["median"]),
            step=1000,
            help="Valuation of the property, vehicle, or assets offered as security."
        )

    with loan_col:
        st.subheader("📝 Loan Information")
        
        loan_amount = st.number_input(
            "Requested Loan Amount ($)",
            min_value=0,
            max_value=5000000,
            value=int(num_ranges["Loan_Amount"]["median"]),
            step=1000,
            help="Total loan capital requested."
        )
        
        loan_term = st.selectbox(
            "Loan Term (Months)",
            options=[12, 24, 36, 48, 60, 72, 84],
            index=[12, 24, 36, 48, 60, 72, 84].index(int(num_ranges["Loan_Term"]["median"])),
            help="Standard repayment duration. Values are matching training terms."
        )
        
        loan_purpose = st.selectbox(
            "Loan Purpose",
            options=categorical_options["Loan_Purpose"]
        )
        
        property_area = st.selectbox(
            "Property Area",
            options=categorical_options["Property_Area"]
        )

# 5. Form Submission & Validation
st.markdown("<br>", unsafe_allow_html=True)
submit_btn = st.button("Assess Loan Eligibility", use_container_width=True)

if submit_btn:
    # Package input dictionary
    inputs = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
        "Employment_Status": employment_status
    }
    
    # Run validation
    validation_errors = validate_inputs(inputs)
    
    if validation_errors:
        st.error("⚠️ **Validation Failed:** Please fix the following errors before prediction.")
        for err in validation_errors:
            st.markdown(f"- {err}")
    else:
        st.success("✅ Inputs validated successfully. Running decision engine...")
        
        # Make Prediction
        try:
            pred_class, approval_prob, rejection_prob = make_prediction(inputs)
            
            st.markdown("<hr style='border-color: #1e3a8a;'>", unsafe_allow_html=True)
            st.subheader("Assessment Results")
            
            res_left, res_right = st.columns([1, 1])
            
            with res_left:
                if pred_class == 1:
                    st.markdown("""
                        <div class="prediction-eligible">
                            <h3 style="margin-top: 0; color: #065f46;">✅ Likely Eligible</h3>
                            <p style="margin: 0; font-size: 0.95rem;">
                                Based on model evaluation, this applicant profile satisfies standard risk criteria. 
                                Further verification of documentation is recommended.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    pred_label = "Likely Eligible"
                else:
                    st.markdown("""
                        <div class="prediction-review">
                            <h3 style="margin-top: 0; color: #78350f;">⚠️ Further Review Recommended</h3>
                            <p style="margin: 0; font-size: 0.95rem;">
                                Based on model evaluation, the applicant's risk factors are elevated. 
                                Manual underwriting review of financial capacities is advised.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    pred_label = "Further Review Recommended"
                
                # Confidence Score
                if approval_prob >= 0.85 or rejection_prob >= 0.85:
                    confidence = "High"
                    conf_color = "green"
                elif approval_prob >= 0.70 or rejection_prob >= 0.70:
                    confidence = "Moderate"
                    conf_color = "orange"
                else:
                    confidence = "Low / Uncertain"
                    conf_color = "red"
                    
                st.markdown(f"**Model Confidence Level:** <span style='color: {conf_color}; font-weight: bold;'>{confidence}</span>", unsafe_allow_html=True)
                
                # Summary Table
                st.markdown("<br><b>Applicant Metrics Summary</b>", unsafe_allow_html=True)
                summary_df = pd.DataFrame({
                    "Metric": ["Household Income", "Requested Loan", "Credit Score", "DTI Ratio", "Liquid Savings", "Collateral Coverage"],
                    "Value": [
                        f"${applicant_income + coapplicant_income:,.2f}",
                        f"${loan_amount:,.2f}",
                        f"{credit_score}",
                        f"{dti_ratio:.2f}",
                        f"${savings:,.2f}",
                        f"${collateral_value:,.2f}"
                    ]
                })
                st.table(summary_df)
                
            with res_right:
                # Plotly Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = approval_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Model Approval Probability", 'font': {'size': 18, 'color': '#1e3a8a'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1e3a8a"},
                        'bar': {'color': "#1d4ed8"},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "#cbd5e1",
                        'steps': [
                            {'range': [0, 40], 'color': '#fee2e2'},
                            {'range': [40, 70], 'color': '#fef3c7'},
                            {'range': [70, 100], 'color': '#d1fae5'}
                        ],
                        'threshold': {
                            'line': {'color': "#ef4444", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(
                    height=240, 
                    margin=dict(l=30, r=30, t=50, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#64748b", 'family': "Outfit"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Rule-based Explanatory Observations
                st.markdown("<b>Risk & Strength Explanatory Observations</b>", unsafe_allow_html=True)
                
                obs = []
                # 1. Credit Profile
                if credit_score >= 740:
                    obs.append("🟢 **Strong Credit Profile**: FICO score is excellent, showing high repayment reliability.")
                elif credit_score < 600:
                    obs.append("🔴 **Substandard Credit Profile**: FICO score suggests elevated credit default risk.")
                    
                # 2. DTI Ratio
                if dti_ratio >= 0.45:
                    obs.append("🔴 **High DTI Ratio**: Elevated debt-service ratio suggests tight repayment bandwidth.")
                elif dti_ratio <= 0.25:
                    obs.append("🟢 **Low DTI Ratio**: Low leverage buffer supports additional loan payment load.")
                    
                # 3. Loan to Income
                total_income = applicant_income + coapplicant_income
                if total_income > 0 and loan_amount > 3 * total_income:
                    obs.append("🟡 **High Loan-to-Income**: Loan amount exceeds three times annual household income.")
                    
                # 4. Savings Coverage
                if savings >= 0.5 * loan_amount:
                    obs.append("🟢 **Strong Savings Coverage**: liquid assets exceed 50% of loan size.")
                elif savings < 0.1 * loan_amount:
                    obs.append("🟡 **Thin Savings Cushion**: Liquid reserves cover less than 10% of requested debt.")
                    
                # 5. Collateral Value
                if collateral_value >= loan_amount:
                    obs.append("🟢 **Full Collateral Asset**: Asset value provides complete security coverage.")
                elif collateral_value > 0 and collateral_value < 0.5 * loan_amount:
                    obs.append("🟡 **Insufficient Collateral**: Asset offers limited coverage (<50%) relative to loan amount.")
                    
                # 6. Coapplicant income
                if coapplicant_income > 2000:
                    obs.append("🟢 **Coapplicant Contribution**: Co-borrower enhances repayment capacity.")
                    
                # 7. Active loan count
                if existing_loans >= 3:
                    obs.append("🟡 **High Active Leverage**: Multiple existing loans indicate a highly utilized credit profile.")
                    
                if not obs:
                    st.write("Profile features are consistent with standard distributions. No major outliers detected.")
                else:
                    for ob in obs:
                        st.markdown(ob)
                st.caption("*Note: Observations are standard rule-based financial indicators separate from ML model predictions.*")
            
            # 8. CSV Report Downloader
            csv_data = io.StringIO()
            csv_writer = csv.writer(csv_data)
            csv_writer.writerow(["Underwriting Metric", "Applicant Value"])
            for k, v in inputs.items():
                csv_writer.writerow([k.replace("_", " "), v])
            csv_writer.writerow([])
            csv_writer.writerow(["Model Recommendation", pred_label])
            csv_writer.writerow(["Approval Probability", f"{approval_prob:.4f}"])
            csv_writer.writerow(["Rejection Probability", f"{rejection_prob:.4f}"])
            csv_writer.writerow(["Confidence Level", confidence])
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Applicant Underwriting Report (CSV)",
                data=csv_data.getvalue(),
                file_name=f"creditwise_assessment_{credit_score}_{int(loan_amount)}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.caption("Please contact system administrators.")
