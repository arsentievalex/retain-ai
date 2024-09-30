import pandas as pd
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import os
import streamlit as st
import base64
from fpdf import FPDF


def get_employee_snapshot(selected_row_df):
    # compose a string about selected employee
    employee_details = f"""
    The employee {selected_row_df['Employee Name'].values[0]} is a {selected_row_df['Role'].values[0]} 
    at {selected_row_df['Department'].values[0]} department with high risk of attrition.
    
    Below is more information about the employee:
    - Tenure: {selected_row_df['Tenure'].values[0]} year(s)
    - Age: {selected_row_df['Age'].values[0]}
    - Years of Experience: {selected_row_df['Years of Experience'].values[0]}
    - Starting Salary: {selected_row_df['Starting Salary'].values[0]}
    - Current Salary: {selected_row_df['Current Salary'].values[0]}
    - Average Monthly Working Hours: {selected_row_df['Average Monthly Working Hours'].values[0]}
    - Last Performance Review Score: {selected_row_df['Last Performance Review Score'].values[0]}
    - Number of Promotions: {selected_row_df['Promotion History'].values[0]}
    - Months in Role: {selected_row_df['Months in Role'].values[0]}
    - Location: {selected_row_df['Location'].values[0]}
    - Contract: {selected_row_df['Contract'].values[0]}
    """

    # get additional data from performance review table
    df_performance = pd.read_csv('/project/data/performance_reviews.csv')

    # filter to keep only selected employee
    df_performance = df_performance[df_performance['Employee ID'] == selected_row_df['Employee ID'].values[0]]

    # Check if there is any data for the selected employee
    if not df_performance.empty:
        # Initialize an empty string to store the performance reviews
        performance_review = "Previous Performance Reviews of the employee:\n"

        # Loop through the available rows of the filtered DataFrame
        for index, row in df_performance.iterrows():
            performance_review += f"""
            - Fiscal Quarter: {row['Fiscal Quarter']}
            - Score: {row['Score']}
            - Summary: {row['Performance Review Summary']}
            """
    else:
        # If no data is found for the employee
        performance_review = "No performance review data available for the selected employee."

    # get additional data from benefits enrollment table
    df_benefits = pd.read_csv('/project/data/benefits_enrollment.csv')

    # filter to keep only selected employee
    df_benefits = df_benefits[df_benefits['Employee ID'] == selected_row_df['Employee ID'].values[0]]

    # compose a string about benefits enrollment
    benefits_enrollment = f"""
    Benefits Enrollment of the employee:
    - Fitness: {df_benefits['Fitness'].values[0]}
    - Insurance: {df_benefits['Insurance'].values[0]}
    - Learning: {df_benefits['Learning'].values[0]}
    - Lunch Card: {df_benefits['Lunch Card'].values[0]}
    """

    # get additional data from engagement survey table
    df_engagement = pd.read_csv('/project/data/engagement_survey.csv')

    # filter to keep only selected employee
    df_engagement = df_engagement[df_engagement['Employee ID'] == selected_row_df['Employee ID'].values[0]]

    # Check if there is any data for the selected employee
    if not df_engagement.empty:
        # Initialize an empty string to store the engagement survey responses
        engagement_survey = "Engagement Survey Responses of the employee:\n"

        # Loop through the available rows of the filtered DataFrame
        for index, row in df_engagement.iterrows():
            engagement_survey += f"""
            - Work-Life Balance Score: {row['Work-Life Balance Score']}
            - Work-Life Balance Comment: {row['Work-Life Balance Comment']}
            - Manager Support Score: {row['Manager Support Score']}
            - Manager Support Comment: {row['Manager Support Comment']}
            - Team Collaboration Score: {row['Team Collaboration Score']}
            - Team Collaboration Comment: {row['Team Collaboration Comment']}
            """
    else:
        # If no data is found for the employee
        engagement_survey = "No engagement survey data available for the selected employee."

    # compose all into a single string
    employee_snapshot = employee_details + "\n" + performance_review + "\n" + benefits_enrollment + "\n" + engagement_survey

    return employee_snapshot


def get_query_engine():
    Settings.text_splitter = SentenceSplitter(chunk_size=256)
    documents = SimpleDirectoryReader("/project/data/pdf_docs").load_data()
    Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

    index = VectorStoreIndex.from_documents(documents)
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct", max_tokens=1024)
    query_engine = index.as_query_engine(similarity_top_k=5, streaming=True)

    return query_engine


# Function to create PDF
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Add text to the PDF
    pdf.multi_cell(0, 10, text)

    # Save the PDF in memory
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output


# Function to download the PDF
def download_pdf(pdf_data, filename):
    b64_pdf = base64.b64encode(pdf_data).decode('latin1')
    download_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Click here to download PDF</a>'
    st.markdown(download_link, unsafe_allow_html=True)
