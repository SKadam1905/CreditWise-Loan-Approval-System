import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from utils.ui_components import init_page, render_metric_card
from utils.prediction import load_metadata, load_pipeline, load_historical_data

# 1. Initialize Page
init_page("Model Performance")

# 2. Main Headers
st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='font-size: 2.25rem; margin-bottom: 0.5rem; color: #1e3a8a;'>📉 Model Performance</h1>
        <h3 style='font-size: 1.1rem; font-weight: 400; color: #475569; margin: 0;'>evaluation metrics, comparisons, and validation visualizations</h3>
    </div>
""", unsafe_allow_html=True)

# 3. Load Metadata & Pipeline
try:
    metadata = load_metadata()
    pipeline = load_pipeline()
    df_raw = load_historical_data()
except Exception as e:
    st.error("⚠️ Model configuration files not found. Please run the model training script (`train_model.py`) first.")
    st.stop()

# 4. Top Metric Cards
st.subheader("Final Selected Model Performance")
st.write(f"The selected model is: **{metadata['model_name']}**")

perf_cols = st.columns(5)
with perf_cols[0]:
    render_metric_card("Accuracy", f"{metadata['accuracy']*100:.2f}%", "🎯")
with perf_cols[1]:
    render_metric_card("Precision", f"{metadata['precision']*100:.2f}%", "🔍")
with perf_cols[2]:
    render_metric_card("Recall (Sensitivity)", f"{metadata['recall']*100:.2f}%", "🛡️")
with perf_cols[3]:
    render_metric_card("F1-Score", f"{metadata['f1_score']*100:.2f}%", "🏆")
with perf_cols[4]:
    render_metric_card("ROC-AUC", f"{metadata['roc_auc']*100:.2f}%", "📈")

st.markdown("<br>", unsafe_allow_html=True)

# 5. Model Comparison
st.subheader("Model Comparison Sweep")
st.write("Results of the 5-fold stratified cross-validation and independent test evaluation across candidate algorithms:")

# Convert comparison dictionary to DataFrame
comparison_dict = metadata["model_comparison"]
comp_rows = []
for m_name, m_metrics in comparison_dict.items():
    comp_rows.append({
        "Model": m_name,
        "CV F1-Score": f"{m_metrics['cv_f1']*100:.2f}%",
        "Test Accuracy": f"{m_metrics['test_accuracy']*100:.2f}%",
        "Test Precision": f"{m_metrics['test_precision']*100:.2f}%",
        "Test Recall": f"{m_metrics['test_recall']*100:.2f}%",
        "Test F1-Score": f"{m_metrics['test_f1']*100:.2f}%",
        "Test ROC-AUC": f"{m_metrics['test_roc_auc']*100:.2f}%" if not np.isnan(m_metrics['test_roc_auc']) else "N/A"
    })
df_comparison = pd.DataFrame(comp_rows)
st.table(df_comparison)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Confusion Matrix & ROC Curve (Generated Dynamically for consistency)
st.subheader("Evaluation Visualizations")
col_chart_left, col_chart_right = st.columns(2)

# Generate test predictions and probabilities dynamically
df_clean = df_raw.dropna(subset=["Loan_Approved"])
df_clean["Loan_Approved"] = df_clean["Loan_Approved"].map({"Yes": 1, "No": 0})
X = df_clean.drop(columns=["Applicant_ID", "Loan_Approved"])
y = df_clean["Loan_Approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
cm = confusion_matrix(y_test, y_pred)

with col_chart_left:
    # 6a. Heatmap Confusion Matrix
    z = cm
    x_lbl = ['Predicted: No', 'Predicted: Yes']
    y_lbl = ['Actual: No', 'Actual: Yes']
    annot = [[str(val) for val in row] for row in z]
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=z, x=x_lbl, y=y_lbl,
        colorscale='Blues',
        text=annot,
        texttemplate="%{text}",
        textfont={"size": 16, "weight": "bold"},
        showscale=False
    ))
    fig_cm.update_layout(
        title="Confusion Matrix (Test Data)",
        height=320,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Outfit"}
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_chart_right:
    # 6b. ROC Curve
    if y_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_score = metadata['roc_auc']
        
        fig_roc = go.Figure()
        # Diagonal reference line
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#94a3b8'), name='Random Guess'))
        # ROC trace
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#1d4ed8', width=3), name=f"ROC (AUC = {auc_score:.4f})"))
        
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            height=320,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(x=0.6, y=0.15),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'family': "Outfit"}
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("ROC Curve unavailable for this model.")

st.markdown("<br>", unsafe_allow_html=True)

# 7. Feature Importance or Coefficients
st.subheader("Global Feature Importance")
st.write(
    "The chart below displays feature importances (or coefficients for linear models) "
    "extracted directly from the trained pipeline. This represents how much each feature "
    "contributed globally to the model's decisions on average."
)

feat_imp_dict = metadata.get("feature_importance", {})
if feat_imp_dict:
    # Sort and take top 15 features
    sorted_feats = sorted(feat_imp_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    df_feat = pd.DataFrame(sorted_feats, columns=["Feature", "Importance/Coefficient"])
    # Sort for horizontal bar chart (ascending order so high values are at the top)
    df_feat = df_feat.sort_values(by="Importance/Coefficient", ascending=True)
    
    # Color bars by positive/negative if they are coefficients, otherwise blue
    if any(v < 0 for v in feat_imp_dict.values()):
        df_feat["Direction"] = df_feat["Importance/Coefficient"].apply(lambda x: "Positive Impact" if x >= 0 else "Negative Impact")
        fig_feat = px.bar(
            df_feat, 
            x="Importance/Coefficient", 
            y="Feature", 
            orientation="h",
            color="Direction",
            color_discrete_map={"Positive Impact": "#10b981", "Negative Impact": "#ef4444"}
        )
    else:
        fig_feat = px.bar(
            df_feat, 
            x="Importance/Coefficient", 
            y="Feature", 
            orientation="h",
            color_discrete_sequence=["#1d4ed8"]
        )
        
    fig_feat.update_layout(
        height=450,
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Outfit"},
        xaxis_title="Relative Score / Coefficient Value",
        yaxis_title=""
    )
    st.plotly_chart(fig_feat, use_container_width=True)
else:
    st.info("Feature importance data is not available for this model type.")

st.markdown("<br>", unsafe_allow_html=True)

# 8. Detailed Classification Report
st.subheader("Detailed Classification Report (Test Data)")
report_dict = classification_report(y_test, y_pred, target_names=["Not Approved", "Approved"], output_dict=True)
df_rep = pd.DataFrame(report_dict).transpose().iloc[:3, :3]
# Format values as percentages
df_rep = df_rep.applymap(lambda x: f"{x*100:.2f}%")
st.table(df_rep)
st.caption("Precision measures exactness, Recall measures completeness, and F1-Score blends both metrics into a single balance indicator.")
