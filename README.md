# Responsible AI – Churn Model Interpretation

Model interpretability and fairness analysis for the Telco Customer Churn prediction model. Built during Internspark AI internship.

## Overview
- **Goal:** Explain model predictions and audit for bias across demographic groups.
- **Model:** Random Forest classifier (from Task 1).
- **Techniques:** SHAP, LIME, fairness metrics (FPR, FNR).

## Key Findings
- **Top predictors:** Tenure, contract type, monthly charges (consistent across SHAP, LIME, and built-in importance).
- **Gender fairness:** No significant disparity — FPR within 0.01, FNR within 0.02.
- **Senior citizens:** Recall slightly higher for seniors (0.561) than non-seniors (0.489) — no negative bias.
- **Primary limitation:** Low overall recall (0.508). Class imbalance leaves nearly half of churners undetected.

## Recommendations
- Apply SMOTE or class weighting to improve recall.
- Monitor FPR/FNR across groups post-deployment with a 0.05 drift threshold.

## Files
- `Task4_Responsible_AI.ipynb` — Full notebook with SHAP, LIME, and bias analysis.

## Tools
Python, Scikit-learn, SHAP, LIME, Pandas, Matplotlib, Google Colab

## Author
[Alan Raju](https://www.linkedin.com/in/alan-raju-09bb0040a/) | [GitHub](https://github.com/zom1xo)
