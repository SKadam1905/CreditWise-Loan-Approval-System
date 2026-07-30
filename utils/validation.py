from typing import Dict, List, Tuple

def validate_inputs(data: Dict[str, any]) -> List[str]:
    """
    Validates applicant inputs and returns a list of error messages.
    An empty list indicates all inputs are valid.
    """
    errors = []

    # 1. Applicant & Coapplicant Income
    app_income = data.get("Applicant_Income")
    if app_income is None:
        errors.append("Applicant Income is required.")
    elif app_income < 0:
        errors.append("Applicant Income cannot be negative.")
    elif app_income < 1000:
        errors.append("Applicant Income seems unrealistically low (must be at least $1,000).")

    coapp_income = data.get("Coapplicant_Income")
    if coapp_income is not None and coapp_income < 0:
        errors.append("Coapplicant Income cannot be negative.")

    # 2. Loan Amount
    loan_amount = data.get("Loan_Amount")
    if loan_amount is None:
        errors.append("Loan Amount is required.")
    elif loan_amount <= 0:
        errors.append("Loan Amount must be greater than zero.")
    elif loan_amount < 500:
        errors.append("Loan Amount is too small (must be at least $500).")

    # 3. Age
    age = data.get("Age")
    if age is None:
        errors.append("Age is required.")
    elif not (18 <= age <= 100):
        errors.append("Age must be a realistic value between 18 and 100.")
    elif not (21 <= age <= 59):
        # We allow it, but we can note that the training data only covers ages 21 to 59.
        # This is a validation boundary.
        pass

    # 4. Debt-to-Income (DTI) Ratio
    dti = data.get("DTI_Ratio")
    if dti is None:
        errors.append("Debt-to-Income (DTI) Ratio is required.")
    elif not (0.0 <= dti <= 1.0):
        errors.append("DTI Ratio must be between 0.0 and 1.0 (0% to 100%).")
    elif not (0.1 <= dti <= 0.6):
        errors.append("DTI Ratio must stay within the dataset's expected scale of 0.1 to 0.6.")

    # 5. Credit Score
    credit_score = data.get("Credit_Score")
    if credit_score is None:
        errors.append("Credit Score is required.")
    elif not (300 <= credit_score <= 850):
        errors.append("Credit Score must stay within the standard FICO range of 300 to 850.")
    elif not (550 <= credit_score <= 800):
        # The dataset contains scores from 550 to 799. Let's warn or restrict.
        # We can restrict to 300 to 850 as a valid range, but highlight in UI if it's out of training bounds.
        pass

    # 6. Loan Term
    loan_term = data.get("Loan_Term")
    if loan_term is None:
        errors.append("Loan Term is required.")
    elif loan_term <= 0:
        errors.append("Loan Term must be greater than zero.")
    # Standard loan terms from dataset are 12, 24, 36, 48, 60, 72, 84
    elif loan_term not in [12, 24, 36, 48, 60, 72, 84]:
        errors.append("Loan Term must be one of the standard values (12, 24, 36, 48, 60, 72, or 84 months).")

    # 7. Savings & Collateral
    savings = data.get("Savings")
    if savings is not None and savings < 0:
        errors.append("Savings cannot be negative.")

    collateral = data.get("Collateral_Value")
    if collateral is not None and collateral < 0:
        errors.append("Collateral Value cannot be negative.")

    # 8. Dependents & Existing Loans
    dependents = data.get("Dependents")
    if dependents is not None and (dependents < 0 or dependents > 15):
        errors.append("Number of Dependents must be between 0 and 15.")

    existing_loans = data.get("Existing_Loans")
    if existing_loans is not None and existing_loans < 0:
        errors.append("Existing Loans cannot be negative.")

    return errors
