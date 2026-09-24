# Customer Churn Prediction System

An enterprise-grade Machine Learning and Business Intelligence web application that predicts, understands, and mitigates customer churn. 

This system uses a pipeline-based **Logistic Regression** model trained on the Telco Customer Churn dataset and wraps it in a **Streamlit** multi-page dashboard.

---

## 📁 Project Folder Structure

```
Customer_Churn_Prediction/
│
├── .streamlit/
│      └── config.toml           # Streamlit UI Theme configuration (Dark Theme)
│
├── pages/
│      ├── Dashboard.py         # Business intelligence metrics and interactive plots
│      ├── Data_Analysis.py     # Exploratory Data Analysis (EDA) and distributions
│      └── Prediction.py        # Single-customer form and batch CSV scoring
│
├── images/
│      └── logo.png             # ChurnEngine AI sidebar logo
│
├── app.py                      # Router file & Home/About page views
├── train_model.py              # Script to clean data, train, and save model
├── customer_churn.csv          # Local Telco dataset
├── churn_model.pkl             # Serialized Scikit-Learn Pipeline (preprocessor + model)
├── model_metrics.json          # Cached training performance JSON
├── requirements.txt            # Python environment dependencies
├── utils.py                    # Modular helpers (cleaning, styling, model loading)
└── README.md                   # Project documentation (this file)
```
# Customer Churn Prediction

An end-to-end Machine Learning application that predicts whether a customer is likely to churn based on customer information.

## 🚀 Live Demo

[Click here to view the Live Application](YOUR_RENDER_LINK)

## 📌 Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave their service. This project uses Machine Learning to predict customer churn based on historical customer data.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Git & GitHub

## 🤖 Machine Learning

The project uses:

- Logistic Regression
- Data preprocessing
- One-Hot Encoding
- Standard Scaling
- Class Weight Balancing

## 🔄 Project Workflow

Customer Data  
↓  
Data Cleaning  
↓  
Exploratory Data Analysis  
↓  
Feature Engineering  
↓  
Data Preprocessing  
↓  
Model Training  
↓  
Model Evaluation  
↓  
Customer Churn Prediction


## 💻 Application Features

- Customer churn prediction
- Single customer prediction
- Batch CSV prediction
- Data visualization
- Model performance analysis

## 📂 Project Structure

```text
Customer-churn-prediction/
│
├── app.py
├── train_model.py
├── utils.py
├── pages/
├── data/
├── requirements.txt
├── churn_model.pkl
└── README.md
---

## 🛠️ Installation & Setup

Follow these simple steps to run the Customer Churn Prediction System locally:

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Install all required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Train the Logistic Regression model, evaluate its performance, and save the serialized model and stats:
```bash
python train_model.py
```
*This command creates the files `churn_model.pkl` and `model_metrics.json` in your workspace.*

### 3. Launch the Streamlit Web Application
Run the Streamlit server:
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 🧠 Model Architecture & Training Results

Our predictive engine uses a Scikit-Learn **Pipeline** consisting of:
1. **Numerical Preprocessing**: Standard scaling (`StandardScaler`) applied to `tenure`, `MonthlyCharges`, and `TotalCharges`.
2. **Categorical Preprocessing**: One-hot encoding (`OneHotEncoder`) with the first category dropped to avoid multicollinearity.
3. **Classifier**: Regularized **Logistic Regression** with class-balanced weighting (`class_weight='balanced'`) to optimize recall (sensitivity) for high-risk cohorts.

### Performance Summary (Test Set)
- **Accuracy**: 73.81%
- **Precision**: 50.43%
- **Recall (Sensitivity)**: 78.34%
- **F1-Score**: 61.36%
- **ROC AUC**: 0.8417

*Since identifying potential churners is the main business objective, the model is configured to optimize **Recall**, catching over 78% of customers who actually churn.*

---

## 🎨 Premium UI Features
* **Modern Dark Theme**: Configured default dark palette (`backgroundColor = "#0b0d10"`, `primaryColor = "#8A2BE2"`) in `.streamlit/config.toml`.
* **Glassmorphism Metrics**: Translucent card styles with subtle borders, gradients, and micro-animations on hover.
* **Smart Fields**: Single prediction forms automatically disable and autofill irrelevant categories based on phone and internet choices.
* **Interactive Cohort Analytics**: Dynamic visualizations (histograms, pie charts, scatter plots, line charts, and boxplots) powered by Matplotlib and Seaborn.
* **Batch Prediction**: Score entire databases in seconds by dropping a CSV file and exporting predictions.
