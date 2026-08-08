import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Load dataset
df = utils.load_and_clean_data("customer_churn.csv")

# Title and Subtitle
st.markdown('<h1 class="gradient-title">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="gradient-subtitle">Inspect the structure, statistics, and distributions within the Telco Churn dataset.</p>', unsafe_allow_html=True)

# Create a temporary copy of the dataset for readable plotting labels
plot_df = df.copy()
if 'Churn' in plot_df.columns:
    plot_df['Churn'] = plot_df['Churn'].map({0: 'Retained', 1: 'Churned'})

# Tabbed Layout for structured investigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Dataset Summary", 
    "📊 Feature Distributions", 
    "🔍 Categorical Cohort Analysis", 
    "🔗 Correlation Matrix"
])

# --- Tab 1: Dataset Summary ---
with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📁 Data Dimensions & Structure")
    
    col_shape1, col_shape2, col_shape3 = st.columns(3)
    col_shape1.metric("Row Count (Customers)", f"{df.shape[0]:,}")
    col_shape2.metric("Column Count (Features)", f"{df.shape[1]}")
    # Calculate missing cells
    missing_count = df.isnull().sum().sum()
    col_shape3.metric("Missing Values", f"{missing_count}")
    
    st.markdown("#### Feature List & Data Types")
    # Build columns list with data types and missing values counts
    summary_info = pd.DataFrame({
        'Data Type': df.dtypes.astype(str),
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(summary_info, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔢 Numerical Summary Statistics")
    st.dataframe(df.describe().T.style.format(precision=2), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 2: Feature Distributions ---
with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Target Distribution: Churn vs. Retention")
    
    col_pie, col_pie_desc = st.columns([3, 2])
    with col_pie:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor('#0b0d10')
        ax.set_facecolor('#161b22')
        
        churn_counts = plot_df['Churn'].value_counts()
        ax.pie(
            churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
            startangle=90, colors=['#8A2BE2', '#ff007f'],
            textprops=dict(color='#f0f6fc', weight='bold', fontsize=11)
        )
        ax.axis('equal')
        st.pyplot(fig)
    with col_pie_desc:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("""
        The pie chart highlights the class distribution:
        - **Retained (No)**: ~73.5% of the customer cohort.
        - **Churned (Yes)**: ~26.5% of the customer cohort.
        
        This indicates a class imbalance that is standard for customer databases. We handle this imbalance using `class_weight='balanced'` in our predictive engine.
        """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📅 Tenure Distribution")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0b0d10')
        ax.set_facecolor('#161b22')
        
        sns.histplot(data=plot_df, x='tenure', hue='Churn', multiple='dodge', palette={'Retained': '#8A2BE2', 'Churned': '#ff007f'}, bins=20, ax=ax)
        ax.set_xlabel('Tenure (Months)', color='#f0f6fc', fontweight='bold')
        ax.set_ylabel('Customer Count', color='#f0f6fc', fontweight='bold')
        ax.tick_params(colors='#f0f6fc')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, color='#30363d', linestyle=':', alpha=0.5)
        
        # Legend custom colors
        legend = ax.get_legend()
        if legend:
            legend.set_title('Status')
            for text in legend.get_texts():
                text.set_color('#f0f6fc')
                
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_dist2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("💳 Monthly Charges Distribution")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0b0d10')
        ax.set_facecolor('#161b22')
        
        sns.histplot(data=plot_df, x='MonthlyCharges', hue='Churn', multiple='dodge', palette={'Retained': '#8A2BE2', 'Churned': '#ff007f'}, bins=20, ax=ax)
        ax.set_xlabel('Monthly Charges ($)', color='#f0f6fc', fontweight='bold')
        ax.set_ylabel('Customer Count', color='#f0f6fc', fontweight='bold')
        ax.tick_params(colors='#f0f6fc')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, color='#30363d', linestyle=':', alpha=0.5)
        
        legend = ax.get_legend()
        if legend:
            legend.set_title('Status')
            for text in legend.get_texts():
                text.set_color('#f0f6fc')
                
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 3: Categorical Cohort Analysis ---
with tab3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🏷️ Churn Demographics & Services Breakdown")
    
    col_cat1, col_cat2 = st.columns(2)
    
    with col_cat1:
        st.markdown("#### Contract Type Analysis")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor('#0b0d10')
        ax.set_facecolor('#161b22')
        
        sns.countplot(data=plot_df, x='Contract', hue='Churn', palette={'Retained': '#8A2BE2', 'Churned': '#ff007f'}, ax=ax)
        ax.set_xlabel('Contract Type', color='#f0f6fc')
        ax.set_ylabel('Count', color='#f0f6fc')
        ax.tick_params(colors='#f0f6fc')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, color='#30363d', linestyle=':', alpha=0.5)
        
        legend = ax.get_legend()
        if legend:
            legend.set_title('Status')
            for text in legend.get_texts():
                text.set_color('#f0f6fc')
                
        st.pyplot(fig)
        
    with col_cat2:
        st.markdown("#### Internet Service Analysis")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor('#0b0d10')
        ax.set_facecolor('#161b22')
        
        sns.countplot(data=plot_df, x='InternetService', hue='Churn', palette={'Retained': '#8A2BE2', 'Churned': '#ff007f'}, ax=ax)
        ax.set_xlabel('Internet Service Type', color='#f0f6fc')
        ax.set_ylabel('Count', color='#f0f6fc')
        ax.tick_params(colors='#f0f6fc')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, color='#30363d', linestyle=':', alpha=0.5)
        
        legend = ax.get_legend()
        if legend:
            legend.set_title('Status')
            for text in legend.get_texts():
                text.set_color('#f0f6fc')
                
        st.pyplot(fig)
        
    st.markdown("<br>", unsafe_allow_html=True)
    col_cat3, col_cat4 = st.columns(2)
    
    with col_cat3:
        st.markdown("#### Payment Method Analysis")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor('#0b0d10')
        ax.set_facecolor('#161b22')
        
        sns.countplot(data=plot_df, x='PaymentMethod', hue='Churn', palette={'Retained': '#8A2BE2', 'Churned': '#ff007f'}, ax=ax)
        ax.set_xlabel('Payment Method', color='#f0f6fc')
        ax.set_ylabel('Count', color='#f0f6fc')
        ax.tick_params(colors='#f0f6fc')
        plt.xticks(rotation=15, ha='right')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, color='#30363d', linestyle=':', alpha=0.5)
        
        legend = ax.get_legend()
        if legend:
            legend.set_title('Status')
            for text in legend.get_texts():
                text.set_color('#f0f6fc')
                
        st.pyplot(fig)
        
    with col_cat4:
        st.markdown("#### Boxplots: Numerical Features vs Churn")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor('#0b0d10')
        ax.set_facecolor('#161b22')
        
        # Select boxplot metric
        metric_choice = st.radio("Choose feature for boxplot comparison:", ["MonthlyCharges", "tenure"], horizontal=True)
        
        sns.boxplot(
            data=plot_df, x='Churn', y=metric_choice, 
            palette={'Retained': '#8A2BE2', 'Churned': '#ff007f'}, ax=ax
        )
        ax.set_xlabel('Status', color='#f0f6fc')
        ax.set_ylabel(f"{metric_choice} value", color='#f0f6fc')
        ax.tick_params(colors='#f0f6fc')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, color='#30363d', linestyle=':', alpha=0.5)
        
        st.pyplot(fig)
        
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 4: Correlation Matrix ---
with tab4:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔗 Complete Correlation Heatmap")
    
    # Calculate correlations (need numerical values, so map Churn to binary)
    numeric_df = df.copy()
    if 'Churn' in df.columns and df['Churn'].dtype == object:
        numeric_df['Churn'] = numeric_df['Churn'].map({'Yes': 1, 'No': 0})
        
    # We can also map binary categorical variables to numbers for correlation
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in numeric_df.columns:
            # Sort categories to ensure consistent mapping
            cats = sorted(numeric_df[col].dropna().unique())
            if len(cats) == 2:
                numeric_df[col] = numeric_df[col].map({cats[0]: 0, cats[1]: 1})
                
    corr_df = numeric_df.select_dtypes(include=[np.number])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0b0d10')
    ax.set_facecolor('#161b22')
    
    correlation_matrix = corr_df.corr()
    
    sns.heatmap(
        correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
        annot_kws={'size': 9, 'weight': 'bold'}, ax=ax, cbar=True
    )
    
    ax.tick_params(colors='#f0f6fc')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    st.pyplot(fig)
    st.info("💡 Note: Pearson correlation coefficients measure the linear relationship between encoded variables (-1 to +1).")
    st.markdown("</div>", unsafe_allow_html=True)

utils.render_footer()
