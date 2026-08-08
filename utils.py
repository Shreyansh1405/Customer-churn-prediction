import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@st.cache_data
def load_and_clean_data(file_path):
    """
    Loads and cleans the customer churn dataset.
    Handles missing values in TotalCharges, drops duplicates, and encodes the target.
    """
    df = pd.read_csv(file_path)
    
    # Clean TotalCharges: convert blank spaces to NaN and fill with 0.0 (occurs when tenure is 0)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill NaN with 0 because tenure is 0 for these new customers
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Encode target 'Churn' if it exists in dataset
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def get_features_lists():
    """
    Returns lists of numerical features, categorical features, and the target name.
    """
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    target = 'Churn'
    return num_features, cat_features, target

def get_preprocessing_pipeline():
    """
    Creates and returns the preprocessor ColumnTransformer.
    StandardScaler for numerical features, OneHotEncoder for categorical features.
    """
    num_features, cat_features, _ = get_features_lists()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ],
        remainder='drop'
    )
    return preprocessor

@st.cache_resource
def load_trained_model(model_path='churn_model.pkl'):
    """
    Loads the serialized model pipeline (preprocessor + classifier) and caches it.
    """
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def apply_custom_css():
    """
    Injects custom CSS to improve design aesthetics, establish typography,
    create premium glassmorphism metric cards, and animate buttons.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Font styling override */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Outfit', sans-serif !important;
        }
        
        /* Glassmorphism Metric Cards */
        div[data-testid="stMetricValue"] {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            font-size: 2.2rem !important;
            color: #8A2BE2 !important; /* Purple accent */
        }
        div[data-testid="stMetricLabel"] {
            font-family: 'Outfit', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            color: #a1a1aa !important; /* Dim text */
        }
        div[data-testid="stMetric"] {
            background: rgba(22, 27, 34, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(8px) !important;
            border-radius: 16px !important;
            padding: 20px 24px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-4px) !important;
            border-color: rgba(138, 43, 226, 0.4) !important;
            box-shadow: 0 10px 25px rgba(138, 43, 226, 0.15) !important;
        }
        
        /* Premium Gradient Text for Titles */
        .gradient-title {
            background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        .gradient-subtitle {
            font-size: 1.25rem;
            color: #94a3b8;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Custom Cards for Layout Grid */
        .glass-card {
            background: rgba(22, 27, 34, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Custom Styled Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #8A2BE2 0%, #4a00e0 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.2s ease-in-out !important;
            box-shadow: 0 4px 15px rgba(138, 43, 226, 0.25) !important;
        }
        div.stButton > button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 6px 20px rgba(138, 43, 226, 0.4) !important;
        }
        div.stButton > button:active {
            transform: scale(0.98) !important;
        }
        
        /* Sidebar layout modifications */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Progress bar styling customization */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #8A2BE2, #ff007f) !important;
        }
        
        /* Table Styling */
        div[data-testid="stDataFrame"] {
            background-color: #161b22 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Center Footer */
        .footer {
            text-align: center;
            padding: 25px;
            margin-top: 60px;
            font-size: 0.9rem;
            color: #64748b;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .footer a {
            color: #8A2BE2;
            text-decoration: none;
            font-weight: 500;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

def render_footer():
    """
    Renders a premium footer at the bottom of the Streamlit page.
    """
    st.markdown("""
        <div class="footer">
            <p>Customer Churn Prediction System &copy; 2026 | Built with ❤️ using Streamlit & Scikit-Learn</p>
            <p>Designed for Enterprise Business Analytics</p>
        </div>
    """, unsafe_allow_html=True)
