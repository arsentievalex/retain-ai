import streamlit as st
import pandas as pd
from utils import (
    get_employee_snapshot,
    create_pdf,
    download_pdf,
    rename_and_filter_columns,
    feature_engineering,
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
    loaded_model = load("attrition_model_pipeline.joblib")
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


def display_employee_metrics(df):
    """Display key employee metrics like contract type distribution and attrition risk count.
    
    Args:
        df (pd.DataFrame): DataFrame containing employee data with an 'Attrition Probability' column.
        
    This function displays four key metrics about employees in four columns:
    - Total number of unique employees (subordinates)
    - Count of full-time employees
    - Count of part-time employees
    - Count of employees with high attrition risk (AP > 0.5)
    """
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate various metrics from the employee data
    at_risk = df[df["Attrition Probability"] > 0.5].shape[0]
    full_time = df[df["Contract"] == "Full-time"].shape[0]
    part_time = df[df["Contract"] == "Part-time"].shape[0]

    # Display the metrics using Streamlit's metric component
    col1.metric(label="Subordinates", value=df["Employee ID"].nunique())
    col2.metric(label="Full-time", value=full_time)
    col3.metric(label="Part-time", value=part_time)
    col4.metric(label="At risk of attrition", value=at_risk)


def display_predicted_attrition(df):
    """Display predicted attrition probabilities for each employee and allow user selection for recommendations.
    
    Args:
        df (pd.DataFrame): DataFrame containing employee data with 'Attrition Probability' and 'Status' columns.
        
    This function presents the DataFrame in a tabular format, showing attrition probabilities and risk 
    status. It allows users to select an employee row to generate a personalized retention recommendation.
    """
    # Sort employees by attrition probability in descending order to highlight highest risk
    df = df.sort_values("Attrition Probability", ascending=False)

    # Add a 'Status' column where AP > 0.5 is marked with ❗ (high risk) and AP <= 0.5 with ✅ (low risk)
    df["Status"] = df["Attrition Probability"].apply(lambda x: "❗" if x > 0.5 else "✅")

    # Configure the progress bar display for attrition probability column
    column_config_predictions = {
        "Attrition Probability": st.column_config.ProgressColumn(
            min_value=0, max_value=1, format="%.2f"
        )
    }
    
    # Display the table and enable row selection for recommendation generation
    event = st.dataframe(
        df[["Full Name", "Role", "Attrition Probability", "Status"]],
        column_config=column_config_predictions,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # If a row is selected, display a button to generate a retention recommendation for that employee
    if event.selection and event.selection.rows:
        get_retention_recommendation(df, event.selection.rows[0])


def get_retention_recommendation(df, selected_row_index):
    """Generate and download a personalized retention recommendation PDF for a selected employee.
    
    Args:
        df (pd.DataFrame): DataFrame containing employee data with selected row's details.
        selected_row_index (int): Index of the row selected by the user.
        
    This function allows the user to download a retention recommendation PDF with tailored suggestions 
    for the selected employee based on attrition data.
    """
    selected_row_df = df.iloc[[selected_row_index]]
    employee_name = selected_row_df["Full Name"].values[0]

    # Generate a snapshot of the selected employee's data
    employee_snapshot = get_employee_snapshot(selected_row_df)
    st.session_state["employee_snapshot"] = employee_snapshot

    # Button to initiate retention recommendation generation workflow
    get_rec_button = st.button("Retention Recommendation 🪄")
    if get_rec_button:
        # Run the recommendation workflow asynchronously to avoid blocking the UI
        recommendation = asyncio.run(run_workflow())
        
        # Format and clean up the recommendation text
        recommendation = recommendation.replace("**", "")
        
        # Generate and download the PDF
        pdf_data = create_pdf(recommendation)
        download_pdf(pdf_data, filename=f"Retention Recommendation for {employee_name}.pdf")


def display_attrition_methodology(df_feature_importance):
    """Display the importance of various features used in attrition prediction.
    
    Args:
        df_feature_importance (pd.DataFrame): DataFrame containing feature importance scores.
        
    This function displays a table showing the importance of different features in predicting 
    employee attrition, helping users understand factors influencing attrition predictions.
    """
    column_config_importance = {
        "Importance": st.column_config.ProgressColumn(
            min_value=0, max_value=1, format="%.2f"
        )
    }
    
    # Display feature importance scores in a structured table
    st.dataframe(
        df_feature_importance,
        column_config=column_config_importance,
        hide_index=True,
        use_container_width=True,
    )


# Run the main app
main()


