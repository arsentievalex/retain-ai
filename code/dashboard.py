import streamlit as st
import pandas as pd
from utils import get_employee_snapshot, create_pdf, download_pdf, rename_and_filter_columns, feature_engineering
from workflow import run_workflow
import asyncio
from joblib import dump, load


def main():
    st.title('RetainAI Dashboard')
    
    if "employee_df" not in st.session_state and "employee_mappings" not in st.session_state:
        st.warning("Start with Data Upload tab to upload your files or use sample dataset")
        return

    # in case when user switched sample data toggle multiple times and did not load the data
    elif st.session_state['demo_mode']==False and "employee_mappings" not in st.session_state:
        st.warning("Start with Data Upload tab to upload your files or use sample dataset")
        return
        
    if st.session_state['demo_mode']:
        df = st.session_state["employee_df"]
    else:
        df = rename_and_filter_columns(st.session_state["employee_df"], st.session_state["employee_mappings"])
    
    loaded_model = load('attrition_model_pipeline.joblib')
    
    predictions = loaded_model.predict_proba(df)[:, 1]
    
    df['Attrition Probability'] = predictions
    
    # df = pd.read_csv('/project/data/predicted_ap.csv')
    df_feature_importance = pd.read_csv('/project/data/feature_importance.csv')
    
    # # filter to keep only sales
    # df = df[df['Department'] == 'Sales']
    
    # round to 2 decimals
    df = df.round(2)
    df = df.head(15)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # count employees with AP > 0.5
    at_risk = df[df['Attrition Probability'] > 0.5].shape[0]
    
    # count employees with full-time and part-time contract
    full_time = df[df['Contract'] == 'Full-time'].shape[0]
    part_time = df[df['Contract'] == 'Part-time'].shape[0]
    
    col1.metric(label='Subordinates', value=df['Employee ID'].nunique())
    col2.metric(label='Full-time', value=full_time)
    col3.metric(label='Part-time', value=part_time)
    col4.metric(label='At risk of attrition', value=at_risk)
    
    tab1, tab2 = st.tabs(["Predicted Attrition", "AP Methodology"])
    
    # order df by AP
    df = df.sort_values('Attrition Probability', ascending=False)
    
    # status column, if AP above 0.5 then high risk
    df['Status'] = df['Attrition Probability'].apply(lambda x: '❗' if x > 0.5 else '✅')
    
    column_config_predictions = {
        'Attrition Probability': st.column_config.ProgressColumn(
            min_value=0, max_value=1, format="%.2f")
    }
    
    with tab1:
        event = st.dataframe(df[['Full Name', 'Role', 'Attrition Probability', 'Status']], column_config=column_config_predictions,
                     hide_index=True, use_container_width=True, on_select='rerun', selection_mode='single-row')
    
        # Check if any row is selected
        if event.selection and event.selection.rows:
    
            # Add a button to generate retention recommendation
            get_rec_button = st.button("Retention Recommendation 🪄")
    
            if get_rec_button:
                selected_row_index = event.selection.rows[0]
                selected_row_df = df.iloc[[selected_row_index]]
                employee_name = selected_row_df['Full Name'].values[0]
    
                # compose all information about the selected employee
                employee_snapshot = get_employee_snapshot(selected_row_df)
    
                st.session_state['employee_snapshot'] = employee_snapshot
    
                # Run the workflow to generate retention recommendation
                recommendation = asyncio.run(run_workflow())
    
                # Allow user to download the PDF
                recommendation = recommendation.replace("**", "")
                pdf_data = create_pdf(recommendation)
                download_pdf(pdf_data, filename=f"Retention Recommendation for {employee_name}.pdf")
    
    
    with tab2:
        st.subheader("Factors That Influence Attrition Prediction")
        column_config_importance = {
            'Importance': st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.2f")
        }
    
        st.dataframe(df_feature_importance, column_config=column_config_importance, hide_index=True, use_container_width=True)

main()
