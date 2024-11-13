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
    """
    Generates a detailed employee snapshot by composing information from multiple data sources, including
    personal details, performance reviews, benefits enrollment, and engagement survey responses.

    Parameters:
        selected_row_df (DataFrame): A DataFrame containing information about the selected employee.
    
    Returns:
        str: A formatted string summarizing the employee's details, including role, department, 
             tenure, salary history, performance reviews, benefits enrollment, and engagement survey responses.
    """
    
    # Compose basic details about the employee using provided DataFrame columns.
    employee_details = f"""
    The employee {selected_row_df['Full Name'].values[0]} is a {selected_row_df['Role'].values[0]} 
    in the {selected_row_df['Department'].values[0]} department.
    
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

    # Gather performance review details if available
    if "performance reviews_df" in st.session_state:
        df_performance = st.session_state["performance reviews_df"]
        df_performance = df_performance[df_performance['Employee ID'] == selected_row_df['Employee ID'].values[0]]
        
        performance_review = "Previous Performance Reviews of the employee:\n"
        
        # Append each performance review to the string
        for _, row in df_performance.iterrows():
            performance_review += f"""
            - Fiscal Quarter: {row['Fiscal Quarter']}
            - Score: {row['Score']}
            - Summary: {row['Performance Review Summary']}
            """
    else:
        performance_review = "No performance review data available for the selected employee."

    # Gather benefits enrollment details if available
    if "benefits enrollment_df" in st.session_state:
        df_benefits = st.session_state["benefits enrollment_df"]
        df_benefits = df_benefits[df_benefits['Employee ID'] == selected_row_df['Employee ID'].values[0]]
        
        benefits_enrollment = "Benefits Enrollment of the employee:\n"
        
        # List each benefit category and enrollment status
        for _, row in df_benefits.iterrows():
            category = row["Category"]
            status = "Enrolled" if row["Enrollment Status"] else "Not Enrolled"
            benefits_enrollment += f"- {category}: {status}\n"
    else:
        benefits_enrollment = "No benefits enrollment data available for the selected employee."

    # Gather engagement survey responses if available
    if "engagement survey_df" in st.session_state:
        df_engagement = st.session_state["engagement survey_df"]
        df_engagement = df_engagement[df_engagement['Employee ID'] == selected_row_df['Employee ID'].values[0]]
        
        engagement_survey = "Engagement Survey Responses of the employee:\n"
        
        # Append each survey question, score, and comment
        for _, row in df_engagement.iterrows():
            question = row["Question"]
            score = row["Score"]
            comment = row["Comment"]
            engagement_survey += f"- {question}: Score = {score}, Comment = \"{comment}\"\n"
    else:
        engagement_survey = "No engagement survey data available for the selected employee."

    # Combine all sections into a single formatted string
    employee_snapshot = employee_details + "\n" + performance_review + "\n" + benefits_enrollment + "\n" + engagement_survey
    
    return employee_snapshot


@st.cache_resource(show_spinner=False)
def get_query_engine():
    """
    Initializes and returns a query engine for retrieval-augmented generation (RAG) using uploaded or sample PDF documents.
    Sets up document embeddings and the query engine with an NVIDIA language model.
    
    Returns:
        QueryEngine: A query engine configured with embeddings and large language model (LLM) for document-based queries.
    """
    # Configure text splitter settings for chunking text into manageable pieces
    Settings.text_splitter = SentenceSplitter(chunk_size=256)

    # Set demo mode to false if not explicitly provided
    if 'demo_mode' not in st.session_state:
        st.session_state['demo_mode'] = False

    # Load documents from sample or uploaded files based on demo mode setting
    if st.session_state['demo_mode']:
        documents = SimpleDirectoryReader("/project/data/sample_pdf").load_data()
    else:
        documents = SimpleDirectoryReader("/project/data/uploaded_pdf").load_data()

    # Load embedding model for question-answering capabilities
    Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

    # Create a document index for similarity-based retrieval
    index = VectorStoreIndex.from_documents(documents)
    
    # Configure the large language model (LLM) for generating responses
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct", max_tokens=1024)
    
    # Initialize the query engine with top-K similarity search and streaming enabled
    query_engine = index.as_query_engine(similarity_top_k=5, streaming=True)

    return query_engine


@st.cache_resource(show_spinner=False)
def get_chat_engine(context):
    """
    Initializes and returns a chat engine using NVIDIA's large language model (LLM).
    
    Parameters:
        context (str): Initial context for the chat model to provide more tailored responses.
    
    Returns:
        NVIDIA: A pre-configured LLM instance for interactive chat responses.
    """
    # Create an LLM instance with a stable temperature setting for more controlled responses
    llm = NVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0, streaming=True)
    return llm


def create_pdf(text):
    """
    Creates a PDF document from the provided text.

    Parameters:
        text (str): The text content to include in the PDF.
    
    Returns:
        bytes: PDF file data in bytes for download or further processing.
    """
    # Initialize PDF with auto page break and set font
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Add the provided text content to the PDF, supporting multi-line content
    pdf.multi_cell(0, 10, text)

    # Return the PDF as byte data
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output


def download_pdf(pdf_data, filename):
    """
    Creates a download link for a PDF file using base64 encoding.

    Parameters:
        pdf_data (bytes): Byte data of the PDF file.
        filename (str): The name for the downloaded file.
    
    Returns:
        None: Displays a clickable download link in Streamlit.
    """
    # Encode PDF data as base64 for safe embedding in HTML
    b64_pdf = base64.b64encode(pdf_data).decode('latin1')
    
    # Generate the HTML link for downloading the PDF
    download_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Click here to download PDF</a>'
    st.markdown(download_link, unsafe_allow_html=True)


def feature_engineering(df):
    """
    Performs feature engineering on the provided DataFrame by calculating new metrics related to salary,
    tenure, and promotion history. This prepares the data for input to an attrition prediction model.

    Parameters:
        df (DataFrame): The input DataFrame containing employee data.
    
    Returns:
        DataFrame: The transformed DataFrame with additional features and cleaned columns.
    """
    # Drop columns irrelevant to model training or prediction
    df = df.drop(['Full Name', 'ID', 'Start Date', 'End Date'], axis=1, errors='ignore')
    
    # Calculate percentage change in salary from starting to current salary
    df['Salary Percentage Change'] = (df['Current Salary'] - df['Starting Salary']) / df['Starting Salary']
 
    # Salary Raise Per Year calculation (handles zero tenure)
    # Avoid division by zero by adding a small value where Tenure is zero
    epsilon = 1e-6
    df['Adjusted Tenure'] = df['Tenure'].apply(lambda x: x if x > 0 else epsilon)
    df['Salary Raise Per Year'] = (df['Current Salary'] - df['Starting Salary']) / df['Adjusted Tenure']

    # Calculate promotion frequency over tenure period
    df['Promotion Frequency'] = df['Promotion History'] / df['Adjusted Tenure']

    # Drop the temporary Adjusted Tenure column
    df = df.drop('Adjusted Tenure', axis=1)

    return df
