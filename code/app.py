import streamlit as st
import pandas as pd
from utils import get_employee_snapshot, create_pdf, download_pdf
from workflow import run_workflow
import asyncio


st.title('RetainAI Dashboard')

df = pd.read_csv('/project/data/predicted_ap.csv')
df_feature_importance = pd.read_csv('/project/data/feature_importance.csv')

# filter to keep only sales
df = df[df['Department'] == 'Sales']

# round to 2 decimals
df = df.round(2)
df = df.head(15)

with st.sidebar:
    st.header('Your Profile')
    st.image("/project/img/stock_profile_img.png", width=120, caption='Mark Johnson Manager')


col1, col2, col3, col4 = st.columns(4)

# count employees with AP > 0.5
at_risk = df[df['Attrition Probability'] > 0.5].shape[0]

# count employees with full-time and part-time contract
full_time = df[df['Contract'] == 'Full-time'].shape[0]
part_time = df[df['Contract'] == 'Part-time'].shape[0]

col1.metric(label='Subordinates', value=df['Employee ID'].nunique(), delta=2)
col2.metric(label='Full-time', value=full_time, delta=1)
col3.metric(label='Part-time', value=part_time, delta=1)
col4.metric(label='At risk of attrition', value=at_risk, delta=2, delta_color='inverse')

tab1, tab2, tab3 = st.tabs(["Predicted Attrition", "AP Methodology", "Referenced Data"])

# order df by AP
df = df.sort_values('Attrition Probability', ascending=False)

# status column, if AP above 0.5 then high risk
df['Status'] = df['Attrition Probability'].apply(lambda x: '❗' if x > 0.5 else '✅')

column_config_predictions = {
    'Attrition Probability': st.column_config.ProgressColumn(
        min_value=0, max_value=1, format="%.2f")
}

with tab1:
    event = st.dataframe(df[['Employee Name', 'Role', 'Attrition Probability', 'Status']], column_config=column_config_predictions,
                 hide_index=True, use_container_width=True, on_select='rerun', selection_mode='single-row')

    # Check if any row is selected
    if event.selection and event.selection.rows:

        # Add a button to generate retention recommendation
        get_rec_button = st.button("Retention Recommendation 🪄")

        if get_rec_button:
            selected_row_index = event.selection.rows[0]
            selected_row_df = df.iloc[[selected_row_index]]
            employee_name = selected_row_df['Employee Name'].values[0]

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

with tab3:
        st.write("Referenced Documents:")
    
        st.write("1. [B2B SaaS Sales HR Trends.pdf](http://localhost:10000/projects/retain-ai/applications/jupyterlab/lab/tree/data/pdf_docs/B2B_SaaS_Sales_HR_Trends.pdf)")
        st.write("2. [NCorp Employee Benefits.pdf](http://localhost:10000/projects/retain-ai/applications/jupyterlab/lab/tree/data/pdf_docs/NCorp_Employee_Benefits.pdf)")

        st.write("Sample DB Schema:")
        st.image('/project/img/db_schema.svg')


    
    

