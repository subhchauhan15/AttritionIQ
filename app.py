import os
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import streamlit as st
# Try to import SHAP, but handle import errors gracefully
try:
    import shap
    SHAP_AVAILABLE = True
except (ImportError, AttributeError) as e:
    SHAP_AVAILABLE = False
    shap = None


MODEL_PATH = "attrition_pipeline.pkl"


@st.cache_resource
def load_artifacts(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    """Load the trained ML pipeline and related artifacts from disk."""
    artifacts = joblib.load(model_path)
    return artifacts


def encode_categoricals_for_inference(
    df: pd.DataFrame,
    categorical_cols: List[str],
    label_encoders: Dict[str, Any],
    target_column: str,
) -> pd.DataFrame:
    """
    Encode categorical columns using the fitted LabelEncoders from training.
    Unseen categories are mapped to -1.
    """
    df_encoded = df.copy()

    for col in categorical_cols:
        # Skip target and columns that are not present in the uploaded data
        if col == target_column or col not in df_encoded.columns:
            continue

        encoder = label_encoders.get(col)
        if encoder is None:
            continue

        mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
        df_encoded[col] = (
            df_encoded[col]
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

    return df_encoded


def preprocess_input_data(
    raw_df: pd.DataFrame,
    categorical_cols: List[str],
    label_encoders: Dict[str, Any],
    top_features: List[str],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the same preprocessing as training:
    - label encode categoricals
    - drop target (if present)
    - select the top correlated features used in training

    Returns:
        X_features: dataframe with the exact features expected by the pipeline
        df_encoded: encoded dataframe (for optional inspection)
    """
    df_encoded = encode_categoricals_for_inference(
        raw_df, categorical_cols, label_encoders, target_column
    )

    features_df = df_encoded.copy()
    if target_column in features_df.columns:
        features_df = features_df.drop(columns=[target_column])

    # Ensure all expected features exist
    missing_cols = [col for col in top_features if col not in features_df.columns]
    for col in missing_cols:
        features_df[col] = 0

    # Reorder columns to match training
    features_df = features_df[top_features]

    return features_df, df_encoded


def predict_attrition(
    pipeline,
    X: pd.DataFrame,
    positive_class_index: int,
    target_encoder,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate class predictions and positive-class probabilities."""
    preds_numeric = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, positive_class_index]
    preds_label = target_encoder.inverse_transform(preds_numeric)
    return preds_label, probs


def get_feature_importance(pipeline, top_features: List[str]) -> pd.DataFrame:
    """Extract feature importances from the RandomForest classifier in the pipeline."""
    clf = pipeline.named_steps.get("classifier")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    importance = clf.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": top_features, "Importance": importance}
    ).sort_values("Importance", ascending=False)
    return feature_importance_df


def get_shap_explainer(pipeline):
    """Create and cache SHAP explainer for the model using session state."""
    if not SHAP_AVAILABLE:
        return None
    # Use session state to cache the explainer
    if "shap_explainer" not in st.session_state:
        classifier = pipeline.named_steps.get('classifier')
        if classifier is None:
            return None
        st.session_state["shap_explainer"] = shap.TreeExplainer(classifier)
    return st.session_state["shap_explainer"]


def calculate_shap_values(pipeline, X_features: pd.DataFrame, explainer=None):
    """Calculate SHAP values for the given features."""
    try:
        if explainer is None:
            explainer = get_shap_explainer(pipeline)
        
        if explainer is None:
            return None, None
        
        scaler = pipeline.named_steps.get('scaler')
        if scaler is None:
            return None, None
        
        # Scale the features
        X_scaled = scaler.transform(X_features)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_scaled)
        
        return shap_values, explainer
    except Exception as e:
        st.error(f"Error calculating SHAP values: {str(e)}")
        return None, None


def get_employee_shap_values(shap_values, employee_idx: int, top_features: List[str]) -> pd.DataFrame:
    """Extract SHAP values for a specific employee."""
    if shap_values is None:
        return pd.DataFrame()
    
    try:
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # Multi-class: shap_values is a list, use positive class (index 1)
            if len(shap_values) > 1:
                employee_shap = shap_values[1][employee_idx]
            else:
                employee_shap = shap_values[0][employee_idx]
        elif hasattr(shap_values, 'shape'):
            if len(shap_values.shape) == 3:
                # Shape (n_samples, n_features, n_classes)
                if shap_values.shape[2] > 1:
                    employee_shap = shap_values[employee_idx, :, 1]
                else:
                    employee_shap = shap_values[employee_idx, :, 0]
            else:
                # Shape (n_samples, n_features)
                employee_shap = shap_values[employee_idx]
        else:
            return pd.DataFrame()
        
        # Ensure it's a 1D array
        employee_shap = np.array(employee_shap).squeeze()
        
        # Handle case where squeeze might return a scalar
        if employee_shap.ndim == 0:
            employee_shap = np.array([employee_shap])
        
        # Ensure length matches
        if len(employee_shap) != len(top_features):
            return pd.DataFrame()
        
        # Create DataFrame
        shap_df = pd.DataFrame({
            "Feature": top_features,
            "SHAP_Value": employee_shap
        }).sort_values("SHAP_Value", key=abs, ascending=False)
        
        return shap_df
    except (IndexError, KeyError, AttributeError) as e:
        return pd.DataFrame()


def render_upload_page(artifacts: Dict[str, Any]) -> None:
    st.header("📂 Upload Dataset")
    st.markdown(
        "Upload a CSV file with the **same structure** as the training dataset "
        "(e.g., `WA_Fn-UseC_-HR-Employee-Attrition.csv`)."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Awaiting CSV upload.")
        return

    raw_df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(), width='stretch')

    with st.spinner("Preprocessing data and generating predictions..."):
        pipeline = artifacts["pipeline"]
        top_features = artifacts["top_features"]
        categorical_cols = artifacts["categorical_cols"]
        label_encoders = artifacts["label_encoders"]
        target_column = artifacts["target_column"]
        positive_class_index = artifacts["positive_class_index"]
        target_encoder = label_encoders[target_column]

        X_features, df_encoded = preprocess_input_data(
            raw_df,
            categorical_cols,
            label_encoders,
            top_features,
            target_column,
        )

        preds_label, probs = predict_attrition(
            pipeline, X_features, positive_class_index, target_encoder
        )

        positive_label = artifacts["positive_class_label"]

        results_df = raw_df.copy()
        results_df["Attrition_Prediction"] = preds_label
        results_df[f"Attrition_Probability_{positive_label}"] = probs

        # Calculate SHAP values
        shap_values, _ = calculate_shap_values(pipeline, X_features)
        
        # Store in session state for other pages
        st.session_state["uploaded_df"] = raw_df
        st.session_state["results_df"] = results_df
        st.session_state["X_features"] = X_features
        st.session_state["shap_values"] = shap_values
        st.session_state["pipeline"] = pipeline
        st.session_state["top_features"] = top_features

    st.success("Predictions generated successfully! Check the **Predictions** section.")


def render_predictions_page() -> None:
    st.header("📊 Predictions")

    if "results_df" not in st.session_state:
        st.info("No predictions available yet. Please upload a dataset first.")
        return

    results_df = st.session_state["results_df"]

    st.subheader("Prediction Summary")
    col1, col2, col3 = st.columns(3)

    total_rows = len(results_df)
    attrition_col = "Attrition_Prediction"

    if attrition_col in results_df.columns:
        leave_count = (results_df[attrition_col] == "Yes").sum()
        stay_count = total_rows - leave_count
    else:
        leave_count = stay_count = 0

    with col1:
        st.metric("Total Employees", total_rows)
    with col2:
        st.metric("Predicted to Leave", int(leave_count))
    with col3:
        st.metric("Predicted to Stay", int(stay_count))

    st.subheader("Prediction Details")
    st.dataframe(results_df, width='stretch')


def render_feature_importance_page(artifacts: Dict[str, Any]) -> None:
    st.header("🧠 Feature Importance")

    pipeline = artifacts["pipeline"]
    top_features = artifacts["top_features"]

    # Use precomputed importance if available, otherwise compute from pipeline
    feature_importance_df = artifacts.get("feature_importance_df")
    if feature_importance_df is None or feature_importance_df.empty:
        feature_importance_df = get_feature_importance(pipeline, top_features)

    if feature_importance_df.empty:
        st.warning("Feature importance information is not available.")
        return

    st.subheader("Top Contributing Features")

    top_n = st.slider("Number of top features to display", 5, min(25, len(feature_importance_df)), 10)
    top_imp = feature_importance_df.head(top_n)

    st.bar_chart(
        top_imp.set_index("Feature")["Importance"],
        width='stretch',
    )

    st.subheader("Full Feature Importance Table")
    st.dataframe(feature_importance_df, width='stretch')


def render_model_performance_page(artifacts: Dict[str, Any]) -> None:
    st.header("📈 Model Performance")
    
    st.markdown("**Model evaluation metrics on test dataset**")
    
    # Display Accuracy
    st.subheader("Accuracy Scores")
    col1, col2 = st.columns(2)
    
    with col1:
        train_accuracy = artifacts.get("train_accuracy", "N/A")
        if isinstance(train_accuracy, (int, float)):
            st.metric("Train Accuracy", f"{train_accuracy:.4f}")
        else:
            st.metric("Train Accuracy", train_accuracy)
    
    with col2:
        test_accuracy = artifacts.get("test_accuracy", "N/A")
        if isinstance(test_accuracy, (int, float)):
            st.metric("Test Accuracy", f"{test_accuracy:.4f}")
        else:
            st.metric("Test Accuracy", test_accuracy)
    
    # Display Confusion Matrix
    st.subheader("Confusion Matrix")
    confusion_matrix_data = artifacts.get("confusion_matrix")
    
    if confusion_matrix_data is not None:
        cm_df = pd.DataFrame(
            confusion_matrix_data,
            index=["Actual: No", "Actual: Yes"],
            columns=["Predicted: No", "Predicted: Yes"]
        )
        
        st.dataframe(cm_df, width='stretch')
        
        # Visualization
        st.write("**Confusion Matrix Visualization**")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_df,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    else:
        st.warning("Confusion matrix data not available.")
    
    # Display Classification Report
    st.subheader("Classification Report")
    classification_report_data = artifacts.get("classification_report")
    
    if classification_report_data is not None:
        st.text(classification_report_data)
        
        # Parse and display as table
        st.write("**Metrics Breakdown**")
        report_lines = classification_report_data.split('\n')
        
        # Extract metrics for each class
        metrics_data = []
        for line in report_lines[2:-3]:  # Skip header and summary lines
            if line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    metrics_data.append({
                        "Class": parts[0],
                        "Precision": float(parts[1]),
                        "Recall": float(parts[2]),
                        "F1-Score": float(parts[3])
                    })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, width='stretch')
            
            # Metrics visualization
            st.write("**Precision vs Recall by Class**")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = range(len(metrics_df))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], metrics_df['Precision'], width, label='Precision')
            ax.bar([i + width/2 for i in x], metrics_df['Recall'], width, label='Recall')
            
            ax.set_xlabel('Class')
            ax.set_ylabel('Score')
            ax.set_title('Precision and Recall by Class')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df['Class'])
            ax.legend()
            ax.set_ylim([0, 1])
            
            st.pyplot(fig)
    else:
        st.warning("Classification report data not available.")


def render_shap_analysis_page(artifacts: Dict[str, Any]) -> None:
    st.header("🔍 SHAP Analysis")
    
    if not SHAP_AVAILABLE:
        st.error("SHAP library is not available. Please install compatible versions of SHAP and NumPy.")
        st.info("Try: `pip install 'numpy<2' shap` or `pip install --upgrade shap opencv-python`")
        return
    
    if "shap_values" not in st.session_state or st.session_state["shap_values"] is None:
        st.info("No SHAP values available yet. Please upload a dataset first.")
        return
    
    if "results_df" not in st.session_state:
        st.info("No predictions available yet. Please upload a dataset first.")
        return
    
    results_df = st.session_state["results_df"]
    shap_values = st.session_state["shap_values"]
    pipeline = st.session_state["pipeline"]
    top_features = st.session_state["top_features"]
    X_features = st.session_state["X_features"]
    
    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** values explain the contribution of each feature "
        "to the model's prediction for individual employees."
    )
    
    # Employee selection
    total_employees = len(results_df)
    if total_employees == 0:
        st.warning("No employees in the dataset.")
        return
    
    st.subheader("Select Employee to Analyze")
    
    employee_idx = st.selectbox(
        "Choose an employee (by row index)",
        range(total_employees),
        format_func=lambda x: f"Employee {x+1} (Row {x})"
    )
    
    # Validate employee index
    if employee_idx >= len(X_features) or employee_idx < 0:
        st.error(f"Invalid employee index: {employee_idx}")
        return
    
    # Get SHAP values for selected employee
    employee_shap_df = get_employee_shap_values(shap_values, employee_idx, top_features)
    
    if employee_shap_df.empty:
        st.warning("Could not calculate SHAP values for this employee.")
        return
    
    # Display employee info
    st.subheader(f"Employee {employee_idx+1} Details")
    col1, col2 = st.columns(2)
    
    with col1:
        if "Attrition_Prediction" in results_df.columns:
            prediction = results_df.iloc[employee_idx]["Attrition_Prediction"]
            st.metric("Prediction", prediction)
    
    with col2:
        prob_col = [col for col in results_df.columns if "Attrition_Probability" in col]
        if prob_col:
            probability = results_df.iloc[employee_idx][prob_col[0]]
            st.metric("Attrition Probability", f"{probability:.2%}")
    
    # SHAP values visualization
    st.subheader("SHAP Feature Contributions")
    st.markdown(
        "**Positive values** push the prediction toward 'Yes' (attrition), "
        "**negative values** push toward 'No' (stay)."
    )
    
    # Top contributing features
    top_n_shap = st.slider(
        "Number of top features to display",
        5,
        min(25, len(employee_shap_df)),
        10,
        key="shap_top_n"
    )
    
    top_shap = employee_shap_df.head(top_n_shap)
    
    # Bar chart
    st.bar_chart(
        top_shap.set_index("Feature")["SHAP_Value"],
        width='stretch',
    )
    
    # Detailed table
    st.subheader("Full SHAP Values Table")
    
    # Add color coding
    def color_shap_value(val):
        if val > 0:
            return 'background-color: #ffcccc'  # Light red for positive
        else:
            return 'background-color: #ccffcc'  # Light green for negative
    
    styled_df = employee_shap_df.style.applymap(
        color_shap_value,
        subset=["SHAP_Value"]
    )
    st.dataframe(styled_df, width='stretch')
    
    # Summary statistics
    st.subheader("SHAP Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    positive_values = employee_shap_df[employee_shap_df['SHAP_Value'] > 0]['SHAP_Value']
    negative_values = employee_shap_df[employee_shap_df['SHAP_Value'] < 0]['SHAP_Value']
    
    with col1:
        max_pos = positive_values.max() if len(positive_values) > 0 else 0.0
        st.metric("Max Positive Impact", f"{max_pos:.4f}")
    with col2:
        max_neg = negative_values.min() if len(negative_values) > 0 else 0.0
        st.metric("Max Negative Impact", f"{max_neg:.4f}")
    with col3:
        total_pos = positive_values.sum() if len(positive_values) > 0 else 0.0
        st.metric("Total Positive Impact", f"{total_pos:.4f}")
    with col4:
        total_neg = negative_values.sum() if len(negative_values) > 0 else 0.0
        st.metric("Total Negative Impact", f"{total_neg:.4f}")


def main() -> None:
    st.set_page_config(
        page_title="Employee Attrition Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Employee Attrition Prediction Dashboard")
    st.markdown(
        "End-to-end workflow: **Upload CSV → Auto Preprocess → Predict Attrition → "
        "Show Results → Explain Feature Importance**."
    )

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ("Upload Dataset", "Predictions", "Model Performance", "Feature Importance", "SHAP Analysis"),
            index=0,
        )

        st.markdown("---")
        st.markdown("**Model Overview**")
        st.caption(
            "RandomForestClassifier + SMOTE + StandardScaler\n\n"
            "Trained once and loaded from `attrition_pipeline.pkl` for fast inference."
        )

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Trained pipeline file `{MODEL_PATH}` not found.\n\n"
            "Please run `attrition.py` once to train the model and save the pipeline."
        )
        return

    with st.spinner("Loading trained pipeline..."):
        artifacts = load_artifacts()

    if page == "Upload Dataset":
        render_upload_page(artifacts)
    elif page == "Predictions":
        render_predictions_page()
    elif page == "Model Performance":
        render_model_performance_page(artifacts)
    elif page == "Feature Importance":
        render_feature_importance_page(artifacts)
    elif page == "SHAP Analysis":
        render_shap_analysis_page(artifacts)


if __name__ == "__main__":
    main()
