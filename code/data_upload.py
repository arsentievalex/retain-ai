import streamlit as st
import pandas as pd
from fuzzywuzzy import process
from streamlit_pdf_viewer import pdf_viewer
import os

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

# Main App
st.title("RetainAI: Data Uploads")
st.session_state["demo_mode"] = st.toggle("Use sample data")  # Toggle to enable sample data mode
st.header("CSV Uploads")

# Define expected columns for each type of data upload
# These are used to map user-uploaded CSV columns to the app's expected structure
expected_columns_sets = {
    "employee": [
        "Employee ID", "Full Name", "Gender", "Age", "Tenure", "Role",
        "Department", "Starting Salary", "Current Salary", "Location",
        "Contract", "Years of Experience", "Average Monthly Working Hours",
        "Months in Role", "Promotion History", "Last Performance Review Score"
    ],
    "benefits": ["Employee ID", "Category", "Enrollment Status"],
    "reviews": ["Employee ID", "Fiscal Quarter", "Score", "Performance Review Summary"],
    "survey": ["Employee ID", "Question", "Score", "Comment"]
}

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
    # Display CSV section title and indicate if it’s uploaded/loaded
    if session_key in st.session_state or st.session_state["demo_mode"]:
        st.subheader(f"{csv_name} ✅")  # Shows checkmark if data is ready
    else:
        st.subheader(f"{csv_name} (required)")

    if st.session_state["demo_mode"]:
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

# Call handle_csv_upload for each required dataset
# Load CSV files for various data categories required by the app

handle_csv_upload(
    "Employee Data", expected_columns_sets["employee"], "employee_mappings", "/project/data/sample_employee_data.csv"
)
handle_csv_upload(
    "Benefits Enrollment", expected_columns_sets["benefits"], "benefits_mappings", "/project/data/sample_benefits_enrollment.csv"
)
handle_csv_upload(
    "Performance Reviews", expected_columns_sets["reviews"], "reviews_mappings", "/project/data/sample_performance_reviews.csv"
)
handle_csv_upload(
    "Engagement Survey", expected_columns_sets["survey"], "survey_mappings", "/project/data/sample_engagement_survey.csv"
)

# PDF Uploads Section
st.divider()
st.header("PDF Uploads")

# Define paths and titles for optional PDF uploads
pdf_file_info = {
    "industry": {
        "title": "Industry Trends",
        "path": "/project/data/sample_pdf/B2B_SaaS_Sales_HR_Trends.pdf"
    },
    "benefits": {
        "title": "Benefits Enrollment",
        "path": "/project/data/sample_pdf/NCorp_Employee_Benefits.pdf"
    }
}

# Display optional PDF files and allow for uploads if not in demo mode
for key, pdf in pdf_file_info.items():
    if st.session_state.get(f"{key}_pdf") or st.session_state["demo_mode"]:
        # Show PDF title with checkmark if loaded
        st.subheader(f"{pdf['title']} ✅")
        # Provide PDF viewer if the file is available
        with st.expander("Expand to see the content"):
            pdf_viewer(pdf["path"], width=700, key=key)  # Display PDF with a viewer
    else:
        # Option for user to upload the PDF if it's not loaded and demo mode is off
        st.subheader(f"{pdf['title']} (optional)")
        with st.expander("Expand to upload"):
            uploaded_pdf = upload_file(f"Choose PDF with {pdf['title'].lower()}", file_type="pdf")
            if uploaded_pdf and not st.session_state.get(f"{key}_pdf"):
                # Save the uploaded PDF to a specified directory
                file_path = os.path.join("/project/data/uploaded_pdf/", uploaded_pdf.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                st.session_state[f"{key}_pdf"] = True  # Mark PDF as uploaded
                st.rerun()  # Refresh the app to show the uploaded PDF



