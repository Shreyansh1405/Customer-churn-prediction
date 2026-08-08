import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import utils

# Load Model Pipeline
model = utils.load_trained_model()

# Page Titles
st.markdown('<h1 class="gradient-title">AI Churn Prediction Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="gradient-subtitle">Score single customer risk parameters or process batch files using our machine learning models.</p>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Prediction model is not loaded. Please make sure that 'churn_model.pkl' exists and 'train_model.py' has been run.")
else:
    tab_single, tab_batch = st.tabs(["👤 Single Customer Prediction", "📂 Batch File Prediction"])
    
    # --- Tab 1: Single Customer Prediction ---
    with tab_single:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📋 Enter Customer Parameters")
        
        with st.form("single_prediction_form"):
            col_dem, col_acc, col_srv = st.columns(3)
            
            with col_dem:
                st.markdown("##### 👥 Demographics")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner (Married)", ["Yes", "No"])
                dependents = st.selectbox("Dependents (Children/Parents)", ["No", "Yes"])
                
            with col_acc:
                st.markdown("##### 💳 Account Information")
                tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=12)
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox(
                    "Payment Method", 
                    [
                        "Electronic check", 
                        "Mailed check", 
                        "Bank transfer (automatic)", 
                        "Credit card (automatic)"
                    ]
                )
                
                # Monthly charges slider
                monthly_charges = st.slider("Monthly Charges ($)", min_value=15.0, max_value=125.0, value=70.0, step=0.25)
                # Auto-calculated default for Total Charges
                est_total = monthly_charges * tenure
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(est_total), step=10.0)
                
            with col_srv:
                st.markdown("##### ☎️ Services & Add-ons")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                
                # Dynamic MultipleLines option
                if phone_service == "No":
                    multiple_lines = "No phone service"
                    st.caption("Multiple Lines: *Disabled (No Phone)*")
                else:
                    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
                    
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                
                # Dynamic Internet Add-on options
                if internet_service == "No":
                    online_security = "No internet service"
                    online_backup = "No internet service"
                    device_protection = "No internet service"
                    tech_support = "No internet service"
                    streaming_tv = "No internet service"
                    streaming_movies = "No internet service"
                    st.caption("Internet Add-ons: *Disabled (No Internet)*")
                else:
                    online_security = st.selectbox("Online Security", ["No", "Yes"])
                    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
            
            submit_btn = st.form_submit_button("Predict Churn Risk")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submit_btn:
            # 1. Structure Inputs
            input_dict = {
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'gender': [gender],
                'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
                'Partner': [partner],
                'Dependents': [dependents],
                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],
                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],
                'Contract': [contract],
                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method]
            }
            
            input_df = pd.DataFrame(input_dict)
            
            # 2. Run Inference
            pred_class = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            
            churn_risk_pct = pred_proba[1]
            retention_pct = pred_proba[0]
            
            # 3. Display Results Beautifully
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🎯 Prediction Output")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                if pred_class == 1:
                    st.error("🚨 HIGH RISK: Customer is predicted to CHURN")
                    st.metric(label="Churn Probability", value=f"{churn_risk_pct:.2%}")
                    st.markdown("**Retention probability:**")
                    st.progress(retention_pct)
                else:
                    st.success("✅ LOW RISK: Customer is predicted to STAY")
                    st.metric(label="Retention Probability", value=f"{retention_pct:.2%}")
                    st.markdown("**Churn probability:**")
                    st.progress(churn_risk_pct)
                    
            with res_col2:
                st.markdown("##### Prediction Summary Profile")
                summary_data = {
                    'Metric': ['Tenure', 'Monthly Billing', 'Estimated Churn Risk', 'Contract Type'],
                    'Value': [f"{tenure} Months", f"${monthly_charges:.2f}", f"{churn_risk_pct:.1%}", contract]
                }
                st.table(pd.DataFrame(summary_data))
                
                # Single customer CSV download
                export_df = input_df.copy()
                export_df['Predicted_Churn'] = ['Yes' if pred_class == 1 else 'No']
                export_df['Churn_Probability'] = [churn_risk_pct]
                
                csv_buffer = io.StringIO()
                export_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download Customer Prediction Profile",
                    data=csv_buffer.getvalue(),
                    file_name=f"customer_prediction_{tenure}m.csv",
                    mime="text/csv"
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
    # --- Tab 2: Batch File Prediction ---
    with tab_batch:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📂 Process Batch Cohort Datasets")
        
        st.write("""
        Upload a customer CSV dataset matching the feature structure of `customer_churn.csv`.
        The system will calculate churn predictions and probability scores in bulk and generate a download link.
        """)
        
        # Guide template columns
        st.markdown("""
        <details>
        <summary>💡 Click to view required CSV columns</summary>
        <p>The uploaded CSV must contain the following features:</p>
        <pre>gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges</pre>
        <p><i>Note: The customerID and Churn columns are optional and will be skipped/recalculated.</i></p>
        </details>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV Cohort File", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Load CSV
                batch_raw = pd.read_csv(uploaded_file)
                
                st.success("File uploaded successfully!")
                st.markdown("##### Raw Dataset Preview")
                st.dataframe(batch_raw.head(5), use_container_width=True)
                
                # Verify required columns
                required_cols = [
                    'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen', 
                    'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
                ]
                
                missing_cols = [col for col in required_cols if col not in batch_raw.columns]
                
                if missing_cols:
                    st.error(f"Missing columns in uploaded dataset: {missing_cols}")
                else:
                    # Clean batch dataset (copy to avoid mutating input view)
                    batch_cleaned = batch_raw.copy()
                    
                    # Clean TotalCharges blank spaces, etc.
                    batch_cleaned['TotalCharges'] = batch_cleaned['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
                    batch_cleaned['TotalCharges'] = pd.to_numeric(batch_cleaned['TotalCharges'], errors='coerce')
                    batch_cleaned['TotalCharges'] = batch_cleaned['TotalCharges'].fillna(0.0)
                    
                    # Make batch predictions
                    st.info("🤖 Processing machine learning classifications...")
                    
                    # Subset features
                    X_batch = batch_cleaned[required_cols]
                    
                    # Predict classes & probabilities
                    batch_preds = model.predict(X_batch)
                    batch_probs = model.predict_proba(X_batch)[:, 1]
                    
                    # Append results
                    batch_raw['Predicted_Churn'] = np.where(batch_preds == 1, 'Yes', 'No')
                    batch_raw['Churn_Probability'] = batch_probs
                    
                    # Render batch metrics
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("### 📊 Scored Cohort Summary")
                    
                    batch_total = len(batch_raw)
                    batch_churns = int(batch_preds.sum())
                    batch_churn_rate = batch_churns / batch_total
                    avg_prob = float(batch_probs.mean())
                    
                    b_col1, b_col2, b_col3 = st.columns(3)
                    b_col1.metric("Total Processed", f"{batch_total:,}")
                    b_col2.metric("Predicted Churned", f"{batch_churns:,}")
                    b_col3.metric("Cohort Churn Rate", f"{batch_churn_rate:.2%}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("##### Scored Dataset Preview (First 10 Rows)")
                    # Reorder columns to show results at the front
                    res_cols = ['Predicted_Churn', 'Churn_Probability']
                    other_cols = [c for c in batch_raw.columns if c not in res_cols]
                    st.dataframe(batch_raw[res_cols + other_cols].head(10), use_container_width=True)
                    
                    # Export scored batch to CSV
                    csv_batch_buffer = io.StringIO()
                    batch_raw.to_csv(csv_batch_buffer, index=False)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Full Scored Dataset (CSV)",
                        data=csv_batch_buffer.getvalue(),
                        file_name="batch_scored_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error parsing batch CSV file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

utils.render_footer()
