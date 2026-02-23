import pandas as pd
from ydata_profiling import ProfileReport
from scipy.stats import chi2_contingency
import numpy as np
from colorama import Fore
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============= WRAP EVERYTHING IN THIS BLOCK =============
if __name__ == "__main__":
    
    df = pd.read_csv(r"dataset\WA_Fn-UseC_-HR-Employee-Attrition.csv")
    print(df.shape)
    print(df.columns)
    # profile = ProfileReport(df, title="EDA Report for HR_Employee_Attrition", explorative=True)
    # profile.to_file("eda_report.html")        
    
    #---------------------Data Cleaning---------------------------
    numeric_cols = df.select_dtypes(include=['int64']).columns.tolist()
    print(len(numeric_cols))
    # print(df[numeric_cols].info())
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(len(categorical_cols))
    print(df[categorical_cols].info())

    print(Fore.LIGHTWHITE_EX + "------------------Chi-Square Tests------------------")
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    categorical_features.remove('Attrition')

    for col in categorical_features:
        table = pd.crosstab(df[col], df['Attrition'])
        chi2, p, dof, expected = chi2_contingency(table)
    
        print(f"\nFeature: {col}")
        print(f"Chi-square value: {chi2:.4f}")
        print(f"p-value: {p:.6f}")
    
    if p < 0.05:
        print("Significant relationship with Attrition")
    else:
        print("No significant relationship")
    
    
    print(Fore.CYAN+"------------------Data encoding------------------")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    y = df['Attrition']
    x = df.drop('Attrition', axis=1)
    print(y.name)
    print(df["Attrition"].value_counts())
    print(df["Department"].value_counts())
    
    print(Fore.LIGHTBLUE_EX+"------------------Data splitting------------------")
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
    print(Fore.LIGHTGREEN_EX+"------------------feature selection + standardization------------------")
    target_corr = x_train.corrwith(y_train).abs().sort_values(ascending=False)
    top_features = target_corr.head(25).index.tolist()
    x_train_selected = x_train[top_features]
    x_test_selected = x_test[top_features]
    print("Shapes match:", x_train_selected.shape, x_test_selected.shape)
    
    print(Fore.YELLOW+"------------------Creating Pipeline------------------")
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=50,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            random_state=42
        ))
    ])
    
    print(Fore.LIGHTCYAN_EX+"------------------Model training (Pipeline)------------------")
    pipeline.fit(x_train_selected, y_train)
    y_train_pred = pipeline.predict(x_train_selected)
    y_test_pred = pipeline.predict(x_test_selected)
    
    print(Fore.LIGHTMAGENTA_EX+"------------------Model accuracy------------------")
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    print("\nClassification Report:")
    cr = classification_report(y_test, y_test_pred)
    print(cr)
    
    print(Fore.LIGHTRED_EX+"------------------feature imptance------------------")
    importance=pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': top_features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10))
    
    print("\nAll Feature Importances:")
    print(feature_importance_df)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'][:15], feature_importance_df['Importance'][:15])
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print(Fore.LIGHTGREEN_EX + "------------------Saving trained pipeline------------------")
    target_column = "Attrition"
    target_encoder = label_encoders[target_column]
    
    if "Yes" in target_encoder.classes_:
        positive_label = "Yes"
    else:
        positive_label = target_encoder.classes_[-1]
    
    positive_encoded = int(target_encoder.transform([positive_label])[0])
    positive_class_index = int(list(pipeline.classes_).index(positive_encoded))
    
    artifacts = {
        "pipeline": pipeline,
        "top_features": top_features,
        "categorical_cols": categorical_cols,
        "label_encoders": label_encoders,
        "target_column": target_column,
        "feature_importance_df": feature_importance_df,
        "positive_class_index": positive_class_index,
        "positive_class_label": positive_label,
        "x_test_selected": x_test_selected,  
        "y_test": y_test,                     
        "y_test_pred": y_test_pred,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "confusion_matrix": cm,
        "classification_report": cr,
    }
    
    model_path = Path("attrition_pipeline.pkl")
    joblib.dump(artifacts, model_path)
    print(f"Saved trained pipeline and artifacts to {model_path.resolve()}")