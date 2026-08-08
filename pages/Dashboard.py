import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

# Load dataset
df = utils.load_and_clean_data("customer_churn.csv")

# Title and Subtitle
st.markdown('<h1 class="gradient-title">Customer Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="gradient-subtitle">High-level Business Intelligence KPIs and customer retention analytics.</p>', unsafe_allow_html=True)

# 1. Calculate KPI Metrics
total_customers = len(df)
total_churn = df['Churn'].sum()
retention_rate = (total_customers - total_churn) / total_customers
avg_monthly = df['MonthlyCharges'].mean()
avg_tenure = df['tenure'].mean()

# 2. Render Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{total_churn:,}")
col3.metric("Retention Rate", f"{retention_rate:.2%}")
col4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
col5.metric("Avg Tenure", f"{avg_tenure:.1f} mos")

st.markdown("<br><br>", unsafe_allow_html=True)

# 3. Interactive Charts Grid

# --- Row 1: Overall Churn & Churn by Contract ---
col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Overall Churn Distribution")
    
    # Pie Chart
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor('#0b0d10')
    ax.set_facecolor('#161b22')
    
    sizes = [total_customers - total_churn, total_churn]
    labels = ['Retained', 'Churned']
    colors = ['#8A2BE2', '#ff007f'] # Purple and neon pink
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=colors,
        textprops=dict(color='#f0f6fc', weight='bold')
    )
    
    # Make autopct labels white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        
    ax.axis('equal')
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with col_c2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📋 Churn Rate by Contract Type")
    
    # Bar Chart
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor('#0b0d10')
    ax.set_facecolor('#161b22')
    
    # Churn rate calculations by Contract type
    contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
    contract_churn['Churn'] = contract_churn['Churn'] * 100 # percentage
    
    # Plotting
    sns.barplot(
        data=contract_churn, x='Contract', y='Churn', 
        palette=['#ff007f', '#a855f7', '#8A2BE2'], ax=ax
    )
    
    ax.set_xlabel('Contract Type', color='#f0f6fc', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)', color='#f0f6fc', fontweight='bold')
    ax.tick_params(colors='#f0f6fc')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.grid(True, color='#30363d', linestyle=':', alpha=0.5)
    
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Row 2: Monthly Charges Distribution & Retention by Tenure ---
col_c3, col_c4 = st.columns(2)

with col_c3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("💳 Monthly Charges Distribution")
    
    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor('#0b0d10')
    ax.set_facecolor('#161b22')
    
    sns.histplot(
        data=df, x='MonthlyCharges', hue='Churn', multiple='stack', 
        palette={0: '#8A2BE2', 1: '#ff007f'}, kde=True, ax=ax
    )
    
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
            if text.get_text() == '0':
                text.set_text('Retained')
            elif text.get_text() == '1':
                text.set_text('Churned')
                
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with col_c4:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📈 Customer Retention by Tenure")
    
    # Line Chart
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor('#0b0d10')
    ax.set_facecolor('#161b22')
    
    # Tenure retention rate
    tenure_retention = df.groupby('tenure')['Churn'].mean().reset_index()
    tenure_retention['Retention'] = (1 - tenure_retention['Churn']) * 100
    
    ax.plot(tenure_retention['tenure'], tenure_retention['Retention'], color='#8A2BE2', lw=3, label='Retention %')
    ax.fill_between(tenure_retention['tenure'], tenure_retention['Retention'], color='#8A2BE2', alpha=0.15)
    
    ax.set_xlabel('Tenure (Months)', color='#f0f6fc', fontweight='bold')
    ax.set_ylabel('Retention Rate (%)', color='#f0f6fc', fontweight='bold')
    ax.tick_params(colors='#f0f6fc')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.grid(True, color='#30363d', linestyle=':', alpha=0.5)
    
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Row 3: Scatter Plot & Heatmap ---
col_c5, col_c6 = st.columns(2)

with col_c5:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("💵 Monthly Charges vs. Total Charges")
    
    # Scatter Plot
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor('#0b0d10')
    ax.set_facecolor('#161b22')
    
    # Downsample points slightly for cleaner visualization rendering
    sample_df = df.sample(n=min(1800, len(df)), random_state=42)
    sns.scatterplot(
        data=sample_df, x='MonthlyCharges', y='TotalCharges', hue='Churn',
        palette={0: '#8A2BE2', 1: '#ff007f'}, alpha=0.6, s=25, ax=ax
    )
    
    ax.set_xlabel('Monthly Charges ($)', color='#f0f6fc', fontweight='bold')
    ax.set_ylabel('Total Charges ($)', color='#f0f6fc', fontweight='bold')
    ax.tick_params(colors='#f0f6fc')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.grid(True, color='#30363d', linestyle=':', alpha=0.5)
    
    legend = ax.get_legend()
    if legend:
        legend.set_title('Status')
        for text in legend.get_texts():
            text.set_color('#f0f6fc')
            if text.get_text() == '0':
                text.set_text('Retained')
            elif text.get_text() == '1':
                text.set_text('Churned')
                
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with col_c6:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔥 Correlation Heatmap")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor('#0b0d10')
    ax.set_facecolor('#161b22')
    
    corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(
        corr_matrix, annot=True, cmap='Purples', fmt='.3f',
        annot_kws={'size': 11, 'weight': 'bold'}, ax=ax, cbar=False
    )
    
    ax.tick_params(colors='#f0f6fc')
    plt.xticks(rotation=15)
    
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

utils.render_footer()
