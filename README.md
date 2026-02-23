# 🧠 HR Employee Attrition Prediction System  
### End-to-End Machine Learning Project with Explainable AI & Deployment
---

## 📌 1. Problem Statement

Employee attrition is a major challenge for organizations. High attrition leads to:

- Increased recruitment and training costs  
- Productivity loss  
- Workforce instability  
- Knowledge drain  

The objective of this project is to build a Machine Learning system that:

- Predicts whether an employee is likely to leave  
- Identifies key factors influencing attrition  
- Provides explainable insights for HR decision-making  
<!-- - Offers an interactive deployment interface   -->

---

## 📊 2. Data Collection

**Dataset Source:** Kaggle – HR Employee Attrition Dataset  

The dataset contains structured HR information such as:

- Demographics (Age, Gender, Marital Status)  
- Job-related features (Department, JobRole, BusinessTravel)  
- Compensation (MonthlyIncome)  
- Experience metrics (YearsAtCompany, TotalWorkingYears)  
- Target Variable: `Attrition (Yes/No)`  

This is a real-world structured dataset commonly used in HR analytics.
---

## 🔎 3. Exploratory Data Analysis (EDA)

Comprehensive EDA was performed using:

- Pandas  
- Matplotlib & Seaborn  
- YData Profiling (Automated EDA Report)  

### Key Analysis Performed:
- Missing value detection  
- Feature distribution analysis  
- Class imbalance detection  
- Correlation heatmaps  
- Department-wise attrition rates  
- Salary vs attrition comparison  

### Statistical Validation

To strengthen feature understanding:
- **Chi-Square Test** → For categorical features  

This ensured statistically meaningful feature selection.

---

## ⚙️ 4. Data Preprocessing

The following preprocessing steps were implemented:

- Label Encoding for categorical variables  
- Feature scaling using StandardScaler  
- Train-test split (70% training, 30% testing)  
- Feature selection using correlation with target  

### Handling Class Imbalance

Employee attrition datasets are typically imbalanced.

To address this:

- **SMOTE (Synthetic Minority Oversampling Technique)** was applied inside the training pipeline.

---

## 🤖 5. Model Development

### Model Used: Random Forest Classifier

**Why Random Forest?**

- Handles non-linear relationships  
- Robust to noise and outliers  
- Performs well on structured/tabular data  
- Provides built-in feature importance  

---

## 🎛 6. Hyperparameter Tuning

The following hyperparameters were optimized:

- Number of estimators  
- Maximum tree depth  
- Minimum samples split  
- Minimum samples leaf  
- Feature selection strategy  

This improved model generalization performance.

---

## 📈 7. Model Evaluation

The model was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 📊 8. Feature Importance
Feature importance was extracted from the trained Random Forest model.

This helped identify:
- Key drivers of attrition  
- Relative contribution of each feature  

Top influential features included:
- JobRole  
- MonthlyIncome  
- YearsAtCompany  
- BusinessTravel  
- MaritalStatus  
---

## 🔍 9. Explainable AI (SHAP)

To enhance transparency and interpretability:

- SHAP (SHapley Additive Explanations) was implemented  
- Individual employee predictions were explained  
- Positive and negative feature contributions were visualized  

This ensures responsible and interpretable AI decision-making.

---

## 🌐 10. Deployment (Streamlit Application)

The trained model was deployed using **Streamlit**.

### Application Features:

- Upload new employee dataset (CSV)  
- Automatic preprocessing  
- Attrition prediction  
- Probability score generation  
- Feature importance visualization  
- Individual SHAP explanation for employees  

This makes the solution interactive and practical for HR teams.

---

## 🛠 11. Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- SHAP  
- Matplotlib / Seaborn  
- Streamlit  
- Joblib  

---

## 📂 12. Project Structure

```
HR-Employee-Attrition/
│
├── dataset/
├── attrition.py              # Training & ML pipeline script
├── shap_analysis.py          # SHAP explanation script
├── app.py                    # Streamlit deployment app
├── attrition_pipeline.pkl    # Saved trained model artifacts
├── eda_report.html           # Automated EDA report
├── requirements.txt
└── README.md
```

---

## 🚀 13. How to Run the Project

### Step 1: Install Dependencies
```
pip install -r requirements.txt
```

### Step 2: Train the Model (Optional)
```
python attrition.py
```
### Step 3: Run the Streamlit App

streamlit run app.py

---

## 🧠 14. Key Learnings

- Combining statistical analysis with machine learning improves model understanding  
- Handling class imbalance is critical in real-world datasets  
- Model interpretability is as important as accuracy  
- Building end-to-end workflows improves production readiness  

---

## 🎯 15. Future Improvements

- Compare with XGBoost / LightGBM  
- Cloud deployment (AWS / Azure / GCP)  

---

## 🏆 Conclusion

This project demonstrates a complete Data Science lifecycle:

- Business understanding  
- Statistical validation  
- Machine learning modeling  
- Class imbalance handling  
- Explainable AI integration  
<!-- - Interactive deployment   -->
