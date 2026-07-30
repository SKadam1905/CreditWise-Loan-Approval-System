# 💳 CreditWise – AI-Powered Loan Approval Prediction System

CreditWise is an end-to-end Machine Learning application that predicts loan approval eligibility using applicant demographic, financial, and loan information.

The project combines a Scikit-learn Machine Learning pipeline with an interactive Streamlit dashboard to help users evaluate loan applications, estimate approval probabilities, visualize model performance, and explore historical lending patterns.

This project was developed for educational purposes to demonstrate the complete lifecycle of a Machine Learning application—from data preprocessing and feature engineering to model training, evaluation, and deployment using Streamlit.

---

## 🎯 Problem Statement
In traditional banking and consumer finance, assessing loan eligibility manually is time-consuming and prone to human inconsistency. Credit underwriters need a reliable, data-driven system to quickly evaluate risk, identify high-probability borrowers, and highlight key credit observations. 

CreditWise solves this by providing a unified Scikit-learn predictive model and an interactive, clean fintech-style Streamlit dashboard. It helps speed up loan triage while providing transparent, rule-based indicators to back up modeling decisions.

---

## ✨ Key Features

- 🤖 **AI-Based Loan Eligibility Prediction** – Predict whether an applicant is likely to receive loan approval based on financial and demographic information.
- 📈 **Approval Probability Gauge** – Interactive Plotly gauge displaying the model's predicted approval probability.
- ⚠️ **Risk Assessment** – Rule-based financial observations highlighting applicant strengths and potential risks.
- 📊 **Interactive Model Performance Dashboard** – ROC Curve, Confusion Matrix, Feature Importance, and Classification Metrics.
- 📉 **Historical Data Insights** – Interactive visualizations for approval rates, income distributions, and applicant characteristics.
- 📄 **Downloadable Prediction Report** – Export applicant assessment as a CSV report.
- 🚀 **Deployment Ready** – Built with Streamlit and optimized for deployment on Streamlit Community Cloud.

---

## 📊 Dataset Information
The system is trained on an underwriting database of **1,000 applicants** containing the following features:
- **Demographics**: `Age`, `Gender`, `Marital_Status`, `Dependents`, `Education_Level`
- **Employment**: `Employment_Status`, `Employer_Category`
- **Financial Profile**: `Applicant_Income`, `Coapplicant_Income`, `Savings`, `Credit_Score` (FICO), `Existing_Loans`, `DTI_Ratio` (Debt-to-Income), `Collateral_Value`
- **Loan Characteristics**: `Loan_Amount`, `Loan_Term` (months), `Loan_Purpose`, `Property_Area`
- **Target Variable**: `Loan_Approved` (`Yes` / `No`)

---

## ⚙️ Machine Learning Workflow & Preprocessing
To prevent data leakage and ensure standard data engineering, CreditWise utilizes a Scikit-learn `ColumnTransformer` inside an end-to-end `Pipeline`:
1. **Target Cleaning**: Rows with missing target values (`Loan_Approved`) are removed. Target values are converted to binary integers (0/1).
2. **Feature Pruning**: `Applicant_ID` is removed from model training to prevent arbitrary indexing bias.
3. **Data Splitting**: Dataset split into 80% training / 20% test sets using **stratified splitting** to maintain class proportions.
4. **Preprocessing ColumnTransformer**:
   - **Numerical columns** (`Age`, `Income`, `Credit Score`, etc.): Imputed using median values, then standardized via `StandardScaler()`.
   - **Categorical columns** (`Employment Status`, `Marital Status`, etc.): Imputed using most frequent values, then encoded via `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` to maintain dense outputs.
5. **Model Fitting**: Pipeline fits the preprocessor and classifier inside a single serialized `joblib` object.

---

## 🤖 Models Compared & Metrics
We compared 5 classification algorithms using 5-fold stratified cross-validation and independent test evaluations:

| Classification Model | CV F1-Score | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier** | **92.70%** | **95.79%** | **95.16%** | **92.19%** | **93.65%** | **98.77%** |
| **Random Forest Classifier** | 89.71% | 93.16% | 94.74% | 84.38% | 89.26% | 98.52% |
| **Logistic Regression** | 77.29% | 88.42% | 86.21% | 78.12% | 82.02% | 92.06% |
| **Gaussian Naive Bayes** | 75.51% | 87.37% | 81.25% | 81.25% | 81.25% | 93.58% |
| **K-Nearest Neighbors** | 71.68% | 80.00% | 76.79% | 67.19% | 71.68% | 84.65% |

**Selected Model:** `Gradient Boosting Classifier` was selected due to its superior test F1-score (93.65%) and high ROC-AUC (98.77%), which are optimal for handling imbalanced risk profiles.

---

## 📂 Project Structure
```text
CreditWise-Loan-Approval-System/
│
├── app.py                         # Landing page and home dashboard
├── train_model.py                 # Core modeling pipeline and metadata generator
├── requirements.txt               # Pinpoint dependencies for deployment
├── README.md                      # Documentation
├── .gitignore                     # Git ignore file
├── runtime.txt                    # Python environment declaration
├── loan_approval_data.csv         # Local historical dataset
│
├── models/
│   ├── loan_approval_pipeline.joblib  # Saved Scikit-learn Pipeline
│   └── model_metadata.json            # Model metrics, bounds, unique options
│
├── pages/
│   ├── 1_Applicant_Assessment.py  # Underwriting form and prediction engine
│   ├── 2_Model_Performance.py     # Interactive model performance metrics
│   └── 3_Data_Insights.py         # Exploratory data visualizations
│
├── utils/
│   ├── __init__.py
│   ├── prediction.py              # Caching data/pipeline loaders and prediction helpers
│   ├── validation.py              # Front-end forms input validation
│   └── ui_components.py          # Unified CSS injector and shared sidebar
│
└── assets/
    └── logo.png                   # Sleek branding logo
```

---

## 💻 Local Installation

1. **Clone the Repository** and navigate to the project directory:
   ```bash
   cd CreditWise-Loan-Approval-System
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Environment**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

### 1. Train the Model (Optional)
To retrain the model and regenerate the metadata files, run:
```bash
python train_model.py
```
*Note: The pipeline and metadata are pre-trained and saved in the repository, so the app will start instantly without running this step.*

### 2. Launch the Streamlit App
Run the Streamlit application:
```bash
streamlit run app.py
```
This opens the app in your default browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

CreditWise is pre-configured for instant deployment on the **Streamlit Community Cloud**:
1. Push the entire `CreditWise-Loan-Approval-System` repository to GitHub. Ensure `app.py`, `requirements.txt`, and the `models/` directory are in the repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**, and select your GitHub repository, branch, and set the main file path to `app.py`.
4. Click **Deploy**. Your app will build and deploy on a public URL.
5. Paste your URL into your GitHub repository details and resume.

---

## 🌐 Live Demo

**Streamlit Application**

https://creditwise-loan-approval-system-af3scncckhxsvf4y3z2grn.streamlit.app/

**GitHub Repository**

https://github.com/SKadam1905/CreditWise-Loan-Approval-System

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.11 |
| Machine Learning | Scikit-learn |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Web Framework | Streamlit |
| Model Storage | Joblib |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## ⚖️ Limitations & Ethical Considerations
- **Historical Data Bias**: This system predicts eligibility based on historical underwriting data. If historical approvals favored certain income levels or property types, the model will reproduce those biases.
- **Academic Scope**: This tool is not connected to real credit bureaus (Equifax, Experian) and does not represent an actual offer of credit.
- **Explainability**: While rule-based observations explain credit profiles logically, they are not structural representations of Gradient Boosting decision trees.

---

## 📝 Resume Project Description

> **CreditWise – AI-Powered Loan Approval Prediction System**
>
> Developed an end-to-end Machine Learning application using Python, Scikit-learn, and Streamlit to predict loan approval eligibility based on applicant demographic, financial, and loan information. Built a complete preprocessing pipeline using ColumnTransformer, One-Hot Encoding, StandardScaler, and Gradient Boosting Classifier. Designed an interactive dashboard for real-time predictions, model performance visualization, and historical data analysis. Deployed the application using Streamlit with GitHub integration.

---

## 👨‍💻 Author

**Shubham Kadam**

B.Tech Information Technology (Honours in Artificial Intelligence & Machine Learning)

Sanjivani College of Engineering, Kopargaon

📌 GitHub: https://github.com/SKadam1905

📧 Email: kadamshubham1905@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/shubham-kadam-09a184209/

---

## 📄 License

This project is developed for educational, learning, and portfolio purposes only.

The application demonstrates Machine Learning concepts for loan approval prediction and should not be used as a substitute for real-world banking or financial decision-making systems.
