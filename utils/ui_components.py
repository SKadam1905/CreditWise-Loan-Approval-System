import streamlit as st
from pathlib import Path
from utils.prediction import load_metadata

def init_page(page_title: str):
    """
    Initializes a Streamlit page with consistent wide layout, title,
    custom fintech CSS, and renders the standard project sidebar.
    """
    # 1. Page Config (Must be called first, but check if we're already configured)
    # We call it directly. It will work if called at the top of each script.
    st.set_page_config(
        page_title=f"CreditWise - {page_title}",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 2. Inject CSS for Fintech styling
    inject_custom_css()
    
    # 3. Render common Sidebar
    render_sidebar()

def inject_custom_css():
    """
    Injects custom CSS to style the Streamlit application as a modern fintech product.
    """
    st.markdown("""
        <style>
        /* General styling */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Metric cards styling */
        .metric-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .metric-card {
            background-color: var(--background-secondary, #f8fafc);
            border: 1px solid var(--border-color, #e2e8f0);
            border-radius: 12px;
            padding: 1.25rem;
            flex: 1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: #64748b;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0f172a;
        }
        
        /* Dark mode overrides */
        @media (prefers-color-scheme: dark) {
            .metric-card {
                background-color: #1e293b;
                border-color: #334155;
            }
            .metric-value {
                color: #f8fafc;
            }
            .metric-label {
                color: #94a3b8;
            }
        }
        
        /* Success/Warning prediction messages */
        .prediction-eligible {
            padding: 1.5rem;
            background-color: rgba(16, 185, 129, 0.1);
            border-left: 5px solid #10b981;
            border-radius: 8px;
            color: #065f46;
            margin-bottom: 1.5rem;
        }
        
        .prediction-review {
            padding: 1.5rem;
            background-color: rgba(245, 158, 11, 0.1);
            border-left: 5px solid #f59e0b;
            border-radius: 8px;
            color: #78350f;
            margin-bottom: 1.5rem;
        }
        
        @media (prefers-color-scheme: dark) {
            .prediction-eligible {
                color: #a7f3d0;
                background-color: rgba(16, 185, 129, 0.15);
            }
            .prediction-review {
                color: #fde68a;
                background-color: rgba(245, 158, 11, 0.15);
            }
        }
        
        /* Headers styling */
        h1, h2, h3 {
            font-weight: 600;
            color: #1e3a8a;
        }
        
        @media (prefers-color-scheme: dark) {
            h1, h2, h3 {
                color: #3b82f6;
            }
        }
        
        /* Button fintech primary color */
        .stButton>button {
            background-color: #1d4ed8;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            border: none;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            background-color: #1e40af;
            box-shadow: 0 4px 12px rgba(29, 78, 216, 0.3);
            transform: scale(1.02);
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """
    Renders the consistent CreditWise sidebar containing application info,
    model metadata, and legal/academic disclaimers.
    """
    with st.sidebar:
        # App Logo
        base_dir = Path(__file__).resolve().parent.parent
        logo_path = base_dir / "assets" / "logo.png"
        if logo_path.exists():
            col_l, col_c, col_r = st.columns([1, 2, 1])
            with col_c:
                st.image(str(logo_path), use_container_width=True)
        
        # App Title & Branding
        st.markdown("""
            <div style='text-align: center; margin-bottom: 1.5rem; margin-top: 0.5rem;'>
                <h1 style='margin: 0; font-size: 2rem; color: #1e3a8a; display: flex; align-items: center; justify-content: center; gap: 0.5rem;'>
                    💳 CreditWise
                </h1>
                <p style='margin: 0; font-size: 0.9rem; color: #64748b;'>AI Loan Decision Engine</p>
            </div>
            <hr style='margin-top: 0; margin-bottom: 1.5rem; border-color: #e2e8f0;'>
        """, unsafe_allow_html=True)
        
        # Project Overview section
        st.subheader("Project Overview")
        st.write(
            "CreditWise uses advanced machine learning pipelines to analyze applicant data "
            "and assist underwriters in assessing creditworthiness and loan eligibility."
        )
        
        # Load Model Metadata
        try:
            metadata = load_metadata()
            
            st.subheader("Model Information")
            st.markdown(f"**Model Name:** `{metadata['model_name']}`")
            st.markdown(f"**Pipeline Version:** `v1.0.0`")
            st.markdown(f"**F1-Score:** `{metadata['f1_score']*100:.2f}%` (Test)")
            st.markdown(f"**ROC-AUC:** `{metadata['roc_auc']*100:.2f}%` (Test)")
            st.markdown(f"**Trained Date:** `{metadata['training_date'].split()[0]}`")
            
        except Exception:
            # Fallback if metadata is not generated yet
            st.warning("Model metadata is currently unavailable. Please run train_model.py first.")
            
        st.markdown("<hr style='margin: 1.5rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
        
        # Disclaimer
        st.subheader("Academic Disclaimer")
        st.info(
            "This application is an academic decision-support project and does not "
            "provide actual loan approval or financial advice."
        )
        
        st.caption("Developed for portfolio demonstration.")

def render_metric_card(label: str, value: str, icon: str = ""):
    """
    Renders a custom styled metric card.
    """
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)
