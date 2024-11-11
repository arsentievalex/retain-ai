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


def rename_and_filter_columns(df, column_mappings):
    """
    Rename DataFrame columns based on a dictionary of mappings and drop columns with 'No mapping available'.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns need to be renamed.
    - column_mappings (dict): A dictionary where keys are original column names and values are new column names.
      Columns with "No mapping available" as the value will be dropped.

    Returns:
    - pd.DataFrame: The DataFrame with renamed columns and filtered to remove unmapped columns.
    """
    # Filter out columns with 'No mapping available' and create a new dictionary for renaming
    filtered_mappings = {original: new for original, new in column_mappings.items() if new != "No mapping available"}
    
    # Rename columns in the DataFrame using the filtered dictionary
    df = df.rename(columns=filtered_mappings)
    
    # Return DataFrame with only the mapped columns
    return df[filtered_mappings.values()]
    

def get_employee_snapshot(selected_row_df):
    # compose a string about selected employee
    employee_details = f"""
    The employee {selected_row_df['Full Name'].values[0]} is a {selected_row_df['Role'].values[0]} 
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

    # get additional data from performance review table if it exists
    if "reviews_df" in st.session_state:
        df_performance = st.session_state["reviews_df"]
    
        # filter to keep only selected employee
        df_performance = df_performance[df_performance['Employee ID'] == selected_row_df['Employee ID'].values[0]]
    
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
        performance_review = "No performance review data available for the selected employee."

    # get additional data from benefits enrollment table if it exists
    if "benefits_df" in st.session_state:
        df_benefits = st.session_state["benefits_df"]
    
        # filter to keep only selected employee
        df_benefits = df_benefits[df_benefits['Employee ID'] == selected_row_df['Employee ID'].values[0]]
    
        # compose a string about benefits enrollment
        benefits_enrollment = "Benefits Enrollment of the employee:\n"
    
        # Append each category with enrollment status to the string
        for _, row in df_benefits.iterrows():
            category = row["Category"]
            status = "True" if row["Enrollment Status"] else "False"
            benefits_enrollment += f"- {category}: {status}\n"
    else:
        benefits_enrollment = "No benefits enrollment data available for the selected employee."

    # get additional data from engagement survey table if it exists
    if "survey_df" in st.session_state:
        df_engagement = st.session_state["survey_df"]
    
        # filter to keep only selected employee
        df_engagement = df_engagement[df_engagement['Employee ID'] == selected_row_df['Employee ID'].values[0]]
    
        # Initialize an empty string to store the engagement survey responses
        engagement_survey = "Engagement Survey Responses of the employee:\n"
        
        # Append each question, score, and comment to the string
        for _, row in df_engagement.iterrows():
            question = row["Question"]
            score = row["Score"]
            comment = row["Comment"]
            engagement_survey += f"- {question}: Score = {score}, Comment = \"{comment}\"\n"
    else:
        engagement_survey = "No engagement survey data available for the selected employee."

    # compose all into a single string
    employee_snapshot = employee_details + "\n" + performance_review + "\n" + benefits_enrollment + "\n" + engagement_survey

    return employee_snapshot


def get_query_engine():
    Settings.text_splitter = SentenceSplitter(chunk_size=256)

    if 'demo_mode' not in st.session_state:
        st.session_state['demo_mode'] = False

    # load sample pdf docs if user did not upload their files
    if st.session_state['demo_mode']:
        documents = SimpleDirectoryReader("/project/data/sample_pdf").load_data()
    else:
        documents = SimpleDirectoryReader("/project/data/uploaded_pdf").load_data()
        
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


# Feature Engineering
def feature_engineering(df):

    # Drop irrelevant columns
    df = df.drop(['Full Name', 'ID', 'Start Date', 'End Date'], axis=1, errors='ignore')
    
    # Salary Percentage Change
    df['Salary Percentage Change'] = (df['Current Salary'] - df['Starting Salary']) / df['Starting Salary']
 
    # Salary Raise Per Year
    # To avoid division by zero, add a small epsilon where Tenure is zero
    epsilon = 1e-6
    df['Adjusted Tenure'] = df['Tenure'].apply(lambda x: x if x > 0 else epsilon)
    df['Salary Raise Per Year'] = (df['Current Salary'] - df['Starting Salary']) / df['Adjusted Tenure']

    # Promotion Frequency
    df['Promotion Frequency'] = df['Promotion History'] / df['Adjusted Tenure']

    # Drop the Adjusted Tenure column as it's no longer needed
    df = df.drop('Adjusted Tenure', axis=1)

    return df
