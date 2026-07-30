import os
import json
import datetime
import platform
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Scikit-learn imports
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def train_and_evaluate():
    # 1. Load Data
    data_path = Path("loan_approval_data.csv")
    if not data_path.exists():
        raise FileNotFoundError("Missing loan_approval_data.csv in the current working directory.")
        
    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")
    
    # 2. Handle missing target values
    df = df.dropna(subset=["Loan_Approved"])
    print(f"Dataset shape after dropping missing targets: {df.shape}")
    
    # Map target column to binary integers
    df["Loan_Approved"] = df["Loan_Approved"].map({"Yes": 1, "No": 0})
    
    # 3. Separate features and target
    X = df.drop(columns=["Applicant_ID", "Loan_Approved"])
    y = df["Loan_Approved"]
    
    # Define feature lists
    numerical_cols = [
        "Applicant_Income", "Coapplicant_Income", "Age", "Dependents", 
        "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings", 
        "Collateral_Value", "Loan_Amount", "Loan_Term"
    ]
    
    categorical_cols = [
        "Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", 
        "Education_Level", "Gender", "Employer_Category"
    ]
    
    # Verify columns exist in X
    assert all(col in X.columns for col in numerical_cols), "Missing numerical columns in dataset"
    assert all(col in X.columns for col in categorical_cols), "Missing categorical columns in dataset"
    
    # Capture unique categorical options (excluding NaNs) for UI generation
    categorical_options = {}
    for col in categorical_cols:
        unique_vals = df[col].dropna().unique().tolist()
        categorical_options[col] = sorted(unique_vals)
        
    # Capture numerical ranges for UI validation and defaults
    numerical_ranges = {}
    for col in numerical_cols:
        non_null_vals = df[col].dropna()
        numerical_ranges[col] = {
            "min": float(non_null_vals.min()),
            "max": float(non_null_vals.max()),
            "mean": float(non_null_vals.mean()),
            "median": float(non_null_vals.median())
        }
        
    # 4. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 5. Define Preprocessing Pipeline
    # Check sklearn version for sparse_output argument compatibility
    sklearn_version = [int(x) for x in sklearn.__version__.split(".")[:2] if x.isdigit()]
    if len(sklearn_version) >= 2 and (sklearn_version[0] > 1 or (sklearn_version[0] == 1 and sklearn_version[1] >= 2)):
        ohe_kwargs = {"sparse_output": False, "handle_unknown": "ignore"}
    else:
        ohe_kwargs = {"sparse": False, "handle_unknown": "ignore"}
        
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**ohe_kwargs))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ])
    
    # 6. Define Models to Compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB()
    }
    
    results = {}
    best_f1 = -1
    best_model_name = None
    best_pipeline = None
    
    print("\n--- Model Evaluation Sweep ---")
    
    for name, clf in models.items():
        # Build full pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # Cross-validation on training set (5-fold stratified cv)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        mean_cv_f1 = cv_scores.mean()
        
        # Fit on training data
        pipeline.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = pipeline.predict(X_test)
        
        # Calculate probabilities for ROC-AUC
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = float("nan")
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"{name}:")
        print(f"  CV F1:      {mean_cv_f1:.4f}")
        print(f"  Test F1:    {f1:.4f}")
        print(f"  Test ROC:   {roc_auc:.4f}")
        print(f"  Test Acc:   {acc:.4f}")
        
        results[name] = {
            "cv_f1": float(mean_cv_f1),
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
            "test_roc_auc": float(roc_auc),
            "confusion_matrix": cm,
            "classification_report": report
        }
        
        # Select best model based primarily on Test F1, breaking ties with ROC-AUC
        if f1 > best_f1 or (f1 == best_f1 and roc_auc > results[best_model_name]["test_roc_auc"]):
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nFinal Selected Model: {best_model_name} (Test F1: {best_f1:.4f})")
    
    # 7. Save Final Pipeline and Metadata
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    pipeline_path = models_dir / "loan_approval_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_path)
    print(f"Saved complete pipeline to {pipeline_path}")
    
    # Class distribution in target
    target_dist_train = y_train.value_counts().to_dict()
    target_dist_test = y_test.value_counts().to_dict()
    target_dist_overall = y.value_counts().to_dict()
    
    # Calculate feature importance/coefficients if available
    feature_importance = {}
    classifier = best_pipeline.named_steps["classifier"]
    
    # Retrieve feature names out of the ColumnTransformer
    transformer = best_pipeline.named_steps["preprocessor"]
    # Get feature names after preprocessing
    try:
        # For sklearn >= 1.0
        feature_names = transformer.get_feature_names_out()
    except AttributeError:
        # Fallback if get_feature_names_out is not available
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        feature_importance = dict(zip(feature_names, [float(v) for v in importances]))
    elif hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]
        feature_importance = dict(zip(feature_names, [float(v) for v in coefficients]))
        
    metadata = {
        "model_name": best_model_name,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_training_records": len(X_train),
        "num_input_features": X_train.shape[1],
        "accuracy": results[best_model_name]["test_accuracy"],
        "precision": results[best_model_name]["test_precision"],
        "recall": results[best_model_name]["test_recall"],
        "f1_score": results[best_model_name]["test_f1"],
        "roc_auc": results[best_model_name]["test_roc_auc"],
        "cross_validation_score": results[best_model_name]["cv_f1"],
        "target_class_distribution": {
            "train": {str(k): int(v) for k, v in target_dist_train.items()},
            "test": {str(k): int(v) for k, v in target_dist_test.items()},
            "overall": {str(k): int(v) for k, v in target_dist_overall.items()}
        },
        "required_input_fields": {
            "numerical": numerical_cols,
            "categorical": categorical_cols
        },
        "numerical_ranges": numerical_ranges,
        "categorical_options": categorical_options,
        "model_comparison": results,
        "feature_importance": feature_importance,
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__
        }
    }
    
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {metadata_path}")

if __name__ == "__main__":
    train_and_evaluate()
