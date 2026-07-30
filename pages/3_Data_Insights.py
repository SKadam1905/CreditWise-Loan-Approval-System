import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.ui_components import init_page
from utils.prediction import load_historical_data

# 1. Initialize Page
init_page("Data Insights")

# 2. Main Headers
st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='font-size: 2.25rem; margin-bottom: 0.5rem; color: #1e3a8a;'>📊 Historical Data Insights</h1>
        <h3 style='font-size: 1.1rem; font-weight: 400; color: #475569; margin: 0;'>exploratory visualization of credit applicants and approval distributions</h3>
    </div>
""", unsafe_allow_html=True)

# 3. Load Dataset
try:
    df = load_historical_data()
except Exception as e:
    st.error(f"⚠️ Dataset file not found: {e}")
    st.stop()

# 4. Define Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Approval Overview", 
    "💰 Income & Loan Distributions", 
    "🛡️ Credit Metrics & Correlations"
])

# Clean target for plotting
df_clean = df.dropna(subset=["Loan_Approved"])

# ==================== TAB 1: APPROVAL OVERVIEW ====================
with tab1:
    st.subheader("Approval Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of Loan Approved
        target_counts = df_clean["Loan_Approved"].value_counts().reset_index()
        target_counts.columns = ["Status", "Count"]
        target_counts["Status"] = target_counts["Status"].map({"Yes": "Approved", "No": "Not Approved"})
        
        fig_pie = px.pie(
            target_counts, 
            values="Count", 
            names="Status",
            title="Overall Loan Approval Ratio",
            color="Status",
            color_discrete_map={"Approved": "#10b981", "Not Approved": "#ef4444"},
            hole=0.4
        )
        fig_pie.update_layout(font={'family': "Outfit"}, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.caption(
            "**Interpretation:** The historical dataset has a significant class imbalance, "
            "with roughly 68.6% of applicant loans not approved and 31.4% approved. "
            "This highlights why F1-Score and ROC-AUC are crucial evaluation metrics rather than accuracy alone."
        )
        
    with col2:
        # Approval rate by Employment Status
        df_emp = df.dropna(subset=["Employment_Status", "Loan_Approved"])
        emp_rates = df_emp.groupby("Employment_Status")["Loan_Approved"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        emp_rates.columns = ["Employment Status", "Approval Rate (%)"]
        emp_rates = emp_rates.sort_values(by="Approval Rate (%)", ascending=False)
        
        fig_emp = px.bar(
            emp_rates,
            x="Employment Status",
            y="Approval Rate (%)",
            title="Approval Rate by Employment Status",
            color_discrete_sequence=["#1d4ed8"]
        )
        fig_emp.update_layout(
            font={'family': "Outfit"}, 
            yaxis_range=[0, 100], 
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_emp, use_container_width=True)
        
        st.caption(
            "**Interpretation:** Approval rates show clear differences across employment categories. "
            "Historically, salaried and self-employed categories tend to receive higher approvals compared to "
            "contractual workers or unemployed individuals, reflecting typical underwriting preferences for stable income."
        )

    st.markdown("<hr style='border-color: #e2e8f0; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Approval rate by Education Level
        df_edu = df.dropna(subset=["Education_Level", "Loan_Approved"])
        edu_rates = df_edu.groupby("Education_Level")["Loan_Approved"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        edu_rates.columns = ["Education Level", "Approval Rate (%)"]
        
        fig_edu = px.bar(
            edu_rates,
            x="Education Level",
            y="Approval Rate (%)",
            title="Approval Rate by Education Level",
            color="Education Level",
            color_discrete_map={"Graduate": "#1e3a8a", "Not Graduate": "#64748b"}
        )
        fig_edu.update_layout(
            font={'family': "Outfit"}, 
            yaxis_range=[0, 100], 
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_edu, use_container_width=True)
        
        st.caption(
            "**Interpretation:** Graduate applicants exhibit higher average approval rates. "
            "Education level may correlate with lifetime earnings potential and job stability, "
            "influencing historical underwriting risk appetites."
        )
        
    with col4:
        # Approval rate by Property Area
        df_prop = df.dropna(subset=["Property_Area", "Loan_Approved"])
        prop_rates = df_prop.groupby("Property_Area")["Loan_Approved"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).reset_index()
        prop_rates.columns = ["Property Area", "Approval Rate (%)"]
        prop_rates = prop_rates.sort_values(by="Approval Rate (%)", ascending=False)
        
        fig_prop = px.bar(
            prop_rates,
            x="Property Area",
            y="Approval Rate (%)",
            title="Approval Rate by Property Area",
            color_discrete_sequence=["#2563eb"]
        )
        fig_prop.update_layout(
            font={'family': "Outfit"}, 
            yaxis_range=[0, 100], 
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_prop, use_container_width=True)
        
        st.caption(
            "**Interpretation:** Applicants located in Semiurban and Urban areas show higher approval rates "
            "compared to Rural properties in this dataset. This might be tied to higher average property values (collateral) "
            "or variations in average regional applicant incomes."
        )

# ==================== TAB 2: INCOME & LOAN DISTRIBUTIONS ====================
with tab2:
    st.subheader("Financial Distributions")
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        # Applicant Income Distribution
        fig_inc = px.histogram(
            df.dropna(subset=["Applicant_Income"]),
            x="Applicant_Income",
            nbins=30,
            title="Applicant Income Distribution",
            color_discrete_sequence=["#1e3a8a"],
            labels={"Applicant_Income": "Annual Income ($)"}
        )
        fig_inc.update_layout(
            font={'family': "Outfit"}, 
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_inc, use_container_width=True)
        st.caption(
            "**Interpretation:** Applicant income shows a right-skewed distribution, with the bulk of "
            "applicants earning between $5,000 and $15,000 annually. Very high incomes represent a long tail of outliers."
        )
        
    with col_dist2:
        # Loan Amount Distribution
        fig_la = px.histogram(
            df.dropna(subset=["Loan_Amount"]),
            x="Loan_Amount",
            nbins=30,
            title="Requested Loan Amount Distribution",
            color_discrete_sequence=["#2563eb"],
            labels={"Loan_Amount": "Loan Amount ($)"}
        )
        fig_la.update_layout(
            font={'family': "Outfit"}, 
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_la, use_container_width=True)
        st.caption(
            "**Interpretation:** Loan amount requests are fairly evenly distributed, with minor peaks "
            "around standard financing amounts (e.g. $10,000, $20,000, and $30,000)."
        )

    st.markdown("<hr style='border-color: #e2e8f0; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Credit Score Distribution by Approval Status
    st.subheader("Credit Score vs Approval Status")
    fig_cs = px.box(
        df_clean,
        x="Loan_Approved",
        y="Credit_Score",
        color="Loan_Approved",
        title="FICO Credit Score Spread by Loan Approval Status",
        color_discrete_map={"Yes": "#10b981", "No": "#ef4444"},
        labels={"Credit_Score": "FICO Credit Score", "Loan_Approved": "Loan Approved Status"}
    )
    fig_cs.update_layout(
        font={'family': "Outfit"}, 
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False
    )
    st.plotly_chart(fig_cs, use_container_width=True)
    st.caption(
        "**Interpretation:** There is a strong relationship between credit score and approval. "
        "The median credit score of approved loans is significantly higher than that of unapproved loans. "
        "This indicates that underwriting guidelines place high emphasis on credit history as a default risk metric."
    )

# ==================== TAB 3: CREDIT METRICS & CORRELATIONS ====================
with tab3:
    st.subheader("Debt and Correlation Structures")
    
    # DTI Ratio versus Approval Box Plot
    fig_dti = px.box(
        df_clean,
        x="Loan_Approved",
        y="DTI_Ratio",
        color="Loan_Approved",
        title="Debt-to-Income (DTI) Ratio Spread by Approval Status",
        color_discrete_map={"Yes": "#10b981", "No": "#ef4444"},
        labels={"DTI_Ratio": "Debt-to-Income Ratio", "Loan_Approved": "Loan Approved"}
    )
    fig_dti.update_layout(
        font={'family': "Outfit"}, 
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False
    )
    st.plotly_chart(fig_dti, use_container_width=True)
    
    st.caption(
        "**Interpretation:** Approved loans tend to cluster at lower DTI ratios. "
        "A high DTI ratio is a strong risk indicator because a larger share of the applicant's income "
        "is already allocated to existing debt service, reducing their ability to absorb new payments."
    )
    
    st.markdown("<hr style='border-color: #e2e8f0; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Correlation Heatmap for Numerical Columns
    st.subheader("Feature Correlations")
    st.write("Linear correlations among numerical underwriting features:")
    
    num_cols = [
        "Applicant_Income", "Coapplicant_Income", "Age", "Dependents", 
        "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings", 
        "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    df_num = df[num_cols].dropna()
    corr_matrix = df_num.corr()
    
    # Create custom Plotly heatmap
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    fig_heat.update_layout(
        title="Numerical Features Correlation Matrix",
        height=500,
        margin=dict(l=80, r=40, t=60, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Outfit", 'size': 10}
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.caption(
        "**Correlation Disclaimer:** This correlation heatmap measures linear associations between "
        "features. Notice that correlation does *not* imply causation. For example, a minor correlation "
        "between age and credit score does not mean getting older directly improves a FICO score; rather, both "
        "could be influenced by longer credit histories and financial maturity over time."
    )
