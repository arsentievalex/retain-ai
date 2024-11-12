import streamlit as st
import pandas as pd
from fuzzywuzzy import process
from streamlit_pdf_viewer import pdf_viewer
import os

def upload_file(file_label="Choose a file", type="csv"):
    """Handle file upload."""
    return st.file_uploader(file_label, type=[type])

def get_best_match(col_name, expected_columns, used_mappings, threshold=80):
    """Find the best match for a column name from the list of expected columns."""
    match = process.extractOne(col_name, expected_columns)
    if match and match[1] >= threshold and match[0] not in used_mappings:
        return match[0]
    return "No mapping available"

def auto_map_columns(df_columns, expected_columns, threshold=80):
    """Automatically map DataFrame columns to expected columns using fuzzy matching."""
    mappings = {}
    used_mappings = []
    
    for col in df_columns:
        mapped_col = get_best_match(col, expected_columns, used_mappings, threshold)
        mappings[col] = mapped_col
        if mapped_col != "No mapping available":
            used_mappings.append(mapped_col)

    return mappings

def render_mapping_ui(df_columns, mappings, expected_columns):
    """Render the UI for mapping columns, allowing manual selection with all options displayed in each selectbox."""
    manual_mappings = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i, col in enumerate(df_columns):
            st.text_input(f'Original column: {col}', value=col, disabled=True)
    
    with col2:
        for i, col in enumerate(df_columns):
            mapped_value = mappings.get(col, "No mapping available")
            
            # Ensure mapped_value is included at the start if it exists
            options = [mapped_value] + [opt for opt in expected_columns if opt != mapped_value]
            
            # Render the selectbox with all options
            selected_option = st.selectbox(f'Mapped column for {col}', options=options, index=0)
            
            # Update the manual_mappings with the selected option for each column
            manual_mappings[col] = selected_option
    
    return manual_mappings


def save_mappings(manual_mappings, mappings, session_key):
    """Save final mappings to session state and ensure 1:1 uniqueness."""
    final_mappings = {col: manual_mappings.get(col, mappings[col]) for col in mappings}
    reverse_mappings = {}
    
    for col, mapped_col in final_mappings.items():
        if mapped_col != "No mapping available":
            reverse_mappings.setdefault(mapped_col, []).append(col)
    
    duplicate_mappings = {k: v for k, v in reverse_mappings.items() if len(v) > 1}
    
    if duplicate_mappings:
        st.warning(f"Warning: The following columns have multiple mappings: {duplicate_mappings}")
    else:
        # Save mappings in session state
        st.session_state[session_key] = final_mappings

@st.dialog(title="Field Mapping Preview", width="large")
def show_column_mapping_interface(df, expected_columns, session_key):
    """Complete UI and processing workflow for mapping columns of a CSV file."""
    df_columns = df.columns
    
    # Step 1: Automatically map columns
    mappings = auto_map_columns(df_columns, expected_columns)
    
    # Step 2: Render UI for manual mapping
    manual_mappings = render_mapping_ui(df_columns, mappings, expected_columns)
    
    # Step 3: Save mappings on button click
    st.write("")
    if st.button("Save"):
        save_mappings(manual_mappings, mappings, session_key)
        st.rerun()
    if session_key in st.session_state:
        st.success('Saved Successfully')


# Main App
st.title('RetainAI: Data Uploads')
st.session_state['demo_mode'] = st.toggle("Use sample data")
st.header("CSV Uploads")

# define expected columns for mapping
expected_columns_1 = ['Employee ID','Full Name', 'Gender', 'Age', 'Tenure', 'Role', 'Department', 'Starting Salary', 'Current Salary', 'Location', 'Contract', 'Years of Experience', 'Average Monthly Working Hours', 'Months in Role', 'Promotion History', 'Last Performance Review Score']
expected_columns_2 = ['Employee ID', 'Category', 'Enrollment Status']
expected_columns_3 = ['Employee ID', 'Fiscal Quarter', 'Score', 'Performance Review Summary']
expected_columns_4 = ['Employee ID', 'Question', 'Score', 'Comment']


# CSV File 1
if "employee_mappings" in st.session_state or st.session_state["demo_mode"]:
    st.subheader("Employee Data ✅")
else:
    st.subheader("Employee Data (required)")

if st.session_state["demo_mode"]:
    with st.expander("Expand to see the data"):
        df1 = pd.read_csv('/project/data/sample_employee_data.csv', sep=None, engine='python')
        st.session_state["employee_df"] = df1
        st.dataframe(df1, hide_index=True)
        
else:
    with st.expander("Expand to upload"):
        uploaded_file_1 = upload_file("Choose CSV file with employee data")
        if uploaded_file_1 and "employee_mappings" not in st.session_state:
            df1 = pd.read_csv(uploaded_file_1, sep=None, engine='python')
            st.session_state["employee_df"] = df1
            show_column_mapping_interface(df1, expected_columns_1, "employee_mappings")


# CSV File 2
if "benefits_mappings" in st.session_state or st.session_state["demo_mode"]:
    st.subheader("Benefits Enrollment ✅")
else:
    st.subheader("Benefits Enrollment (optional)")

if st.session_state["demo_mode"]:
    with st.expander("Expand to see the data"):
        df2 = pd.read_csv('/project/data/sample_benefits_enrollment.csv', sep=None, engine='python')
        st.session_state["benefits_df"] = df2
        st.dataframe(df2, hide_index=True)
else:  
    with st.expander("Expand to upload"):
        uploaded_file_2 = upload_file("Choose CSV file with benefits enrollment data")
        if uploaded_file_2 and "benefits_mappings" not in st.session_state:
            df2 = pd.read_csv(uploaded_file_2, sep=None, engine='python')
            st.session_state["benefits_df"] = df2
            show_column_mapping_interface(df2, expected_columns_2, "benefits_mappings")


# CSV File 3
if "reviews_mappings" in st.session_state or st.session_state["demo_mode"]:
    st.subheader("Performance Reviews ✅")
else:
    st.subheader("Performance Reviews (optional)")

if st.session_state["demo_mode"]:
    with st.expander("Expand to see the data"):
        df3 = pd.read_csv('/project/data/sample_performance_reviews.csv', sep=None, engine='python')
        st.session_state["reviews_df"] = df3
        st.dataframe(df3, hide_index=True)
else:
    with st.expander("Expand to upload"):
        uploaded_file_3 = upload_file("Choose CSV file with performance reviews data")
        if uploaded_file_3 and "reviews_mappings" not in st.session_state:
            df3 = pd.read_csv(uploaded_file_3, sep=None, engine='python')
            st.session_state["reviews_df"] = df3
            show_column_mapping_interface(df3, expected_columns_3, "reviews_mappings")

# CSV File 4
if "survey_mappings" in st.session_state or st.session_state["demo_mode"]:
    st.subheader("Engagement Survey ✅")
else:
    st.subheader("Engagement Survey (optional)")

if st.session_state["demo_mode"]:
    with st.expander("Expand to see the data"):
        df4 = pd.read_csv('/project/data/sample_engagement_survey.csv', sep=None, engine='python')
        st.session_state["survey_df"] = df4
        st.dataframe(df4, hide_index=True) 
else:
    with st.expander("Expand to upload"):
        uploaded_file_4 = upload_file("Choose CSV file with engagement survey data")
        if uploaded_file_4 and "survey_mappings" not in st.session_state:
            df4 = pd.read_csv(uploaded_file_4, sep=None, engine='python')
            st.session_state["survey_df"] = df4
            show_column_mapping_interface(df4, expected_columns_4, "survey_mappings")


st.divider()

st.header("PDF Uploads")

# Initialize session state variables if they don't exist
if "industry_pdf" not in st.session_state:
    st.session_state["industry_pdf"] = False
    st.session_state["benefits_pdf"] = False

if st.session_state["industry_pdf"] or st.session_state["demo_mode"]:
    st.subheader("Industry Trends ✅")
else:
    st.subheader("Industry Trends (optional)")

# PDF File 1
if st.session_state["demo_mode"]:
    with st.expander("Expand to see the content"):
        pdf_viewer('/project/data/sample_pdf/B2B_SaaS_Sales_HR_Trends.pdf', width=700)
        
else:
    with st.expander("Expand to upload"):
        uploaded_pdf_1 = upload_file("Choose PDF with industry trends", type="pdf")
        # if the file is uploaded for the first time
        if uploaded_pdf_1 and not st.session_state["industry_pdf"]:
            # Set the file path for saving
            file_path = os.path.join('/project/data/uploaded_pdf/', uploaded_pdf_1.name)

            # Save the uploaded file to the local directory
            with open(file_path, "wb") as f:
                f.write(uploaded_pdf_1.getbuffer())

            # update session state
            st.session_state["industry_pdf"] = True
            st.rerun()


# PDF File 2
if st.session_state["demo_mode"]:
    st.subheader("Benefits Enrollment ✅")
    with st.expander("Expand to see the content"):
        pdf_viewer('/project/data/sample_pdf/NCorp_Employee_Benefits.pdf', width=700, key='benefits')
else:  
    st.subheader("Benefits Enrollment (optional)")
    with st.expander("Expand to upload"):
        uploaded_pdf_2 = upload_file("Choose PDF with benefits info")
        # if the file is uploaded for the first time
        if uploaded_pdf_2 and not st.session_state["benefits_pdf"]:
            # Set the file path for saving
            file_path = os.path.join('/project/data/uploaded_pdf/', uploaded_pdf_2.name)

            # Save the uploaded file to the local directory
            with open(file_path, "wb") as f:
                f.write(uploaded_pdf_2.getbuffer())

            # update session state
            st.session_state["benefits_pdf"] = True
            st.rerun()
