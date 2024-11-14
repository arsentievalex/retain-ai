import streamlit as st
import pandas as pd
from utils import (
    get_employee_snapshot,
    create_pdf,
    download_pdf,
    rename_and_filter_columns,
    feature_engineering,
    display_employee_metrics,
    display_predicted_attrition,
    get_retention_recommendation,
    display_attrition_methodology
)
from workflow import run_workflow
import asyncio
from joblib import load


def main():
    """Main dashboard function for RetainAI: displays key metrics and employee attrition insights.
    
    This function serves as the primary entry point for the dashboard. It checks if required data 
    is available, then displays employee metrics, predicted attrition probabilities, and allows the 
    user to generate retention recommendations.
    """
    st.title("RetainAI: Dashboard")

    # Check if the employee data and mappings are available in session state, or if demo mode is enabled
    if "employee data_df" not in st.session_state and "employee_mappings" not in st.session_state:
        st.warning("Start with Data Upload tab to upload your files or use sample dataset")
        return
    elif not st.session_state["demo_mode"] and "employee_mappings" not in st.session_state:
        st.warning("Start with Data Upload tab to upload your files or use sample dataset")
        return

    # Load employee data, applying renaming and filtering based on mappings if not in demo mode
    df = (
        st.session_state["employee data_df"]
        if st.session_state["demo_mode"]
        else rename_and_filter_columns(st.session_state["employee data_df"], st.session_state["employee_mappings"])
    )

    # Load the pre-trained attrition model and use it to generate attrition probability predictions
    loaded_model = load("/project/models/attrition_model_pipeline.joblib")
    predictions = loaded_model.predict_proba(df)[:, 1]
    df["Attrition Probability"] = predictions

    # Load feature importance data for the attrition model
    df_feature_importance = pd.read_csv("/project/data/feature_importance.csv")
    df = df.round(2).head(15)  # Round off values to two decimals and limit display to the first 15 rows

    # Display overall employee metrics (full-time/part-time, attrition risk count, etc.)
    display_employee_metrics(df)

    # Set up two tabs: one for attrition predictions and one for methodology explanations
    tab1, tab2 = st.tabs(["Predicted Attrition", "AP Methodology"])
    with tab1:
        display_predicted_attrition(df)
    with tab2:
        st.write("The table below highlights the key factors the model uses to predict employee attrition. Each factor’s importance score shows how much it influences the likelihood of an employee leaving.")
        display_attrition_methodology(df_feature_importance)


# Run the main app
main()