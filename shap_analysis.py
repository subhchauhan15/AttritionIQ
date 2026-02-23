import shap
import joblib

print("Loading trained model...")
artifacts = joblib.load("attrition_pipeline.pkl")

pipeline = artifacts["pipeline"]
x_test_selected = artifacts["x_test_selected"]
top_features = artifacts["top_features"]

print(f" Model loaded successfully")
print(f" Test samples: {len(x_test_selected)}")
print(f" Features: {len(top_features)}")

# SHAP analysis
print("\nExtracting model components...")
classifier = pipeline.named_steps['classifier']
scaler = pipeline.named_steps['scaler']

x_test_scaled = scaler.transform(x_test_selected)

print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(classifier)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(x_test_scaled)

employee_idx = 0
if len(shap_values.shape) == 3:
    # For binary/multi-class, shape (n_samples, n_features, n_classes)
    employee_shap = shap_values[employee_idx, :, 1]  
elif isinstance(shap_values, list):
    employee_shap = shap_values[1][employee_idx]
else:
    employee_shap = shap_values[employee_idx]

employee_shap = employee_shap.squeeze() 

employee_idx = 0
if isinstance(shap_values, list):
    employee_shap = shap_values[1][employee_idx]
else:
    employee_shap = shap_values[employee_idx]

print(f"\nSHAP Feature Contributions for Employee {employee_idx}:")
print("="*55)
for feature, value in zip(top_features, employee_shap):
    print(f"{feature:30s}: {float(value[0]):+.4f}")