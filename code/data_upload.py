import streamlit as st
import pandas as pd
from fuzzywuzzy import process
from streamlit_pdf_viewer import pdf_viewer
import os
from utils import upload_file, get_best_match, auto_map_columns, render_mapping_ui, save_mappings, show_column_mapping_interface, handle_csv_upload

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
