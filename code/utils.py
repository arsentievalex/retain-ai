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
from fuzzywuzzy import process
from streamlit_pdf_viewer import pdf_viewer

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
def get_chat_engine():
    """
    Initializes and returns a chat engine using NVIDIA's large language model (LLM).
    
    Returns:
        NVIDIA: A pre-configured LLM instance for chat responses.
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


def upload_file(file_label="Choose a file", file_type="csv"):
    """
    Display file uploader widget.
    
    Args:
        file_label (str): The label for the uploader widget.
        file_type (str): The file type to accept (e.g., 'csv', 'pdf').
    
    Returns:
        UploadedFile: The uploaded file object.
    """
    return st.file_uploader(file_label, type=[file_type])


def get_best_match(col_name, expected_columns, used_mappings, threshold=80):
    """
    Find the best match for a column name using fuzzy matching.
    
    Args:
        col_name (str): Column name to find a match for.
        expected_columns (list): List of expected column names.
        used_mappings (list): List of already used mappings.
        threshold (int): Minimum matching score to accept.
    
    Returns:
        str: Best match or "No mapping available" if no match found.
    """
    match = process.extractOne(col_name, expected_columns)
    if match and match[1] >= threshold and match[0] not in used_mappings:
        return match[0]
    return "No mapping available"


def auto_map_columns(df_columns, expected_columns, threshold=80):
    """
    Automatically map DataFrame columns to expected columns using fuzzy matching.
    
    Args:
        df_columns (list): List of DataFrame column names.
        expected_columns (list): List of expected column names.
        threshold (int): Minimum matching score to accept.
    
    Returns:
        dict: Mapping of DataFrame columns to expected columns.
    """
    mappings = {}
    used_mappings = []
    
    for col in df_columns:
        mapped_col = get_best_match(col, expected_columns, used_mappings, threshold)
        mappings[col] = mapped_col
        if mapped_col != "No mapping available":
            used_mappings.append(mapped_col)

    return mappings


def render_mapping_ui(df_columns, mappings, expected_columns):
    """
    Render UI for mapping columns, allowing for manual selection.
    
    Args:
        df_columns (list): List of DataFrame column names.
        mappings (dict): Initial column mappings.
        expected_columns (list): List of expected column names.
    
    Returns:
        dict: User-selected mappings.
    """
    manual_mappings = {}
    
    col1, col2 = st.columns(2)
    
    # Display original columns
    with col1:
        for col in df_columns:
            st.text_input(f'Original column: {col}', value=col, disabled=True)
    
    # Display mapping select boxes
    with col2:
        for col in df_columns:
            mapped_value = mappings.get(col, "No mapping available")
            options = [mapped_value] + [opt for opt in expected_columns if opt != mapped_value]
            selected_option = st.selectbox(f'Mapped column for {col}', options=options, index=0)
            manual_mappings[col] = selected_option
    
    return manual_mappings


def save_mappings(manual_mappings, mappings, session_key):
    """
    Save final mappings to session state and check for uniqueness.
    
    Args:
        manual_mappings (dict): User-selected mappings.
        mappings (dict): Auto-generated mappings.
        session_key (str): Session state key for storing mappings.
    """
    final_mappings = {col: manual_mappings.get(col, mappings[col]) for col in mappings}
    reverse_mappings = {}
    
    # Track duplicate mappings
    for col, mapped_col in final_mappings.items():
        if mapped_col != "No mapping available":
            reverse_mappings.setdefault(mapped_col, []).append(col)
    
    duplicate_mappings = {k: v for k, v in reverse_mappings.items() if len(v) > 1}
    
    if duplicate_mappings:
        st.warning(f"Warning: The following columns have multiple mappings: {duplicate_mappings}")
    else:
        st.session_state[session_key] = final_mappings


@st.dialog(title="Field Mapping Preview", width="large")
def show_column_mapping_interface(df, expected_columns, session_key):
    """
    Complete UI and processing workflow for mapping columns of a CSV file.
    
    Args:
        df (DataFrame): Uploaded data as a DataFrame.
        expected_columns (list): List of expected column names.
        session_key (str): Session state key for storing mappings.
    """
    df_columns = df.columns
    mappings = auto_map_columns(df_columns, expected_columns)
    manual_mappings = render_mapping_ui(df_columns, mappings, expected_columns)
    
    if st.button("Save"):
        save_mappings(manual_mappings, mappings, session_key)
        st.rerun()
    
    if session_key in st.session_state:
        st.success('Saved Successfully')


def handle_csv_upload(csv_name, expected_columns, session_key, sample_file_path):
    """Handles CSV data upload or loads sample data for the selected category.
    
    This function manages the upload or display of CSV data files required by the app.
    It either loads a sample dataset if in demo mode or provides an upload interface
    where users can upload their own CSV files. If a file is uploaded, it will prompt
    the user to map columns as per expected columns.

    Args:
        csv_name (str): The descriptive name of the data type, e.g., 'Employee Data'.
        expected_columns (list): List of columns that the uploaded CSV should contain.
        session_key (str): Unique key to store the mapped columns in session state.
        sample_file_path (str): Path to the sample CSV file to load if in demo mode.
    """
    # Check if data is loaded in session or in demo mode
    if session_key in st.session_state or st.session_state.get("demo_mode"):
        st.subheader(f"{csv_name} ✅")  # Show checkmark if data is ready
    else:
        # Display "(required)" only for "Employee Data," others as "(optional)"
        required_text = "(required)" if csv_name == "Employee Data" else "(optional)"
        st.subheader(f"{csv_name} {required_text}")

    if st.session_state.get("demo_mode"):
        # Load and display sample data if demo mode is active
        with st.expander("Expand to see the data"):
            df = pd.read_csv(sample_file_path, sep=None, engine="python")  # Load sample CSV file
            st.session_state[f"{csv_name.lower()}_df"] = df  # Save DataFrame in session state
            st.dataframe(df, hide_index=True)  # Display data table in the app
    else:
        # Allow user to upload a CSV file for the specific category
        with st.expander("Expand to upload"):
            uploaded_file = upload_file(f"Choose CSV file with {csv_name.lower()} data")
            if uploaded_file and session_key not in st.session_state:
                # Load uploaded CSV file and save it in session state
                df = pd.read_csv(uploaded_file, sep=None, engine="python")
                st.session_state[f"{csv_name.lower()}_df"] = df
                # Show interface to map columns to match expected structure
                show_column_mapping_interface(df, expected_columns, session_key)
