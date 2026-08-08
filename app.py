import os
import json
import pandas as pd
import streamlit as st
import utils

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Premium Custom Styling
utils.apply_custom_css()

# 3. Sidebar Header / Logo (Rendered globally on all pages)
if os.path.exists("images/logo.png"):
    st.sidebar.image("images/logo.png", use_container_width=True)
st.sidebar.markdown("<h3 style='text-align: center; color: #8A2BE2; margin-top: -10px;'>ChurnEngine AI</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.85rem; color: #888;'>Enterprise Prediction Platform</p>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

# 4. Define Inline Pages (Home and About)
def show_home():
    """
    Renders a stunning and modern landing page for the application.
    """
    st.markdown('<h1 class="gradient-title">Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtitle">An enterprise-grade ML analytics platform to predict, analyze, and mitigate customer attrition in real time.</p>', unsafe_allow_html=True)
    
    # System overview cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h3 style="color: #8A2BE2;">📊 Interactive KPI Dashboard</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">Monitor essential Business Intelligence metrics at a glance. Review overall retention rates, average customer tenure, and monthly billing totals through sleek interactive charts.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h3 style="color: #8A2BE2;">📈 Deep Exploratory Analysis</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">Dive deep into cohort demographics, contract durations, payment methods, and services. Pinpoint relationships between key attributes and churn patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h3 style="color: #8A2BE2;">🤖 AI Risk Prediction</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">Score individual customer churn risk using our trained Logistic Regression model, or upload list files to process large-scale batch predictions in seconds.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Instruction section
    st.markdown("### 🗺️ Quick Navigation Guide")
    st.info("""
    Use the sidebar menu to navigate between the different views of the system:
    - **Home (current page)**: Overview and system orientation.
    - **Dashboard**: High-level business stats, filters, and operational metrics.
    - **Data Analysis**: Scientific statistical plots, feature correlations, and data summaries.
    - **Prediction**: Perform real-time customer predictions or upload CSV files for batch scoring.
    - **About**: Check model performance metrics, classification reports, and engine methodology details.
    """)
    
    utils.render_footer()

def show_about():
    """
    Renders the Model details and performance evaluation data.
    """
    st.markdown('<h1 class="gradient-title">System Methodology & Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="gradient-subtitle">Detailed documentation of the Machine Learning pipeline, model metrics, and classification reports.</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🧠 Machine Learning Pipeline
    The **ChurnEngine AI** utilizes a modular Scikit-Learn pipeline to prepare raw inputs and make predictions.
    
    - **Standardization**: Numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) are scaled via `StandardScaler`.
    - **Categorical Columns**: String categories are transformed using `OneHotEncoder(drop='first')` to maintain numerical integrity.
    - **Model Optimization**: A `LogisticRegression` classifier is trained with L2 regularization and `class_weight='balanced'` to maximize sensitivity (Recall) for churn detection.
    """)
    
    st.markdown("---")
    
    # Load and show metrics from JSON
    if os.path.exists('model_metrics.json'):
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
            
        st.markdown("### 📈 Model Evaluation Metrics (Test Set)")
        
        # Metric cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        m_col2.metric("Precision", f"{metrics['precision']:.2%}")
        m_col3.metric("Recall (Sensitivity)", f"{metrics['recall']:.2%}")
        m_col4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Classification report layout
        col_report, col_cm = st.columns([3, 2])
        
        with col_report:
            st.markdown("#### 📄 Classification Report")
            report_df = pd.DataFrame(metrics['classification_report_dict']).transpose()
            # Clean dataframe layout
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)
            
        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix")
            cm = metrics['confusion_matrix']
            cm_df = pd.DataFrame(
                cm, 
                index=['Retained (Actual No)', 'Churned (Actual Yes)'], 
                columns=['Predicted No', 'Predicted Yes']
            )
            st.dataframe(cm_df, use_container_width=True)
            
        # Draw ROC Curve if matplotlib is loaded
        st.markdown("#### 📈 ROC Curve (Receiver Operating Characteristic)")
        import matplotlib.pyplot as plt
        
        fpr = metrics['roc_curve']['fpr']
        tpr = metrics['roc_curve']['tpr']
        auc_val = metrics['roc_auc']
        
        fig, ax = plt.subplots(figsize=(8, 4))
        # Dark style for matplotlib matching theme
        fig.patch.set_facecolor('#0b0d10')
        ax.set_facecolor('#161b22')
        
        ax.plot(fpr, tpr, color='#8A2BE2', lw=3, label=f'Logistic Regression (AUC = {auc_val:.4f})')
        ax.plot([0, 1], [0, 1], color='#64748b', lw=2, linestyle='--')
        
        ax.set_xlabel('False Positive Rate', color='#f0f6fc')
        ax.set_ylabel('True Positive Rate', color='#f0f6fc')
        ax.set_title('ROC Curve', color='#f0f6fc', fontsize=14, fontweight='bold')
        
        ax.tick_params(colors='#f0f6fc')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            
        ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')
        ax.grid(True, color='#30363d', linestyle=':', alpha=0.6)
        
        st.pyplot(fig)
        
    else:
        st.warning("Model metrics not found. Run 'train_model.py' to generate performance statistics.")
        
    utils.render_footer()

# 5. Define App Navigation
home_page = st.Page(show_home, title="Home", icon="🏠", default=True)
dashboard_page = st.Page("pages/Dashboard.py", title="Dashboard", icon="📊")
analysis_page = st.Page("pages/Data_Analysis.py", title="Data Analysis", icon="📈")
prediction_page = st.Page("pages/Prediction.py", title="Prediction", icon="🤖")
about_page = st.Page(show_about, title="About", icon="ℹ️")

# Run navigation
pg = st.navigation({
    "System Nav": [home_page, dashboard_page, analysis_page, prediction_page, about_page]
})
pg.run()
