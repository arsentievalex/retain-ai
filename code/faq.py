import streamlit as st


st.title("FAQ: How to Use RetainAI")


with st.expander("1. How do I start using the app?"):
    st.write("""
    You can start using the tool in two ways:
    
    - **Option 1:** Upload your own data in CSV and PDF formats. The CSV uploads must follow speficic structure (see next section for a list of required columns).
    - **Option 2:** Use our sample data to test the tool. Select *"Use Sample Data"* to see how the tool processes information and provides predictions based on preloaded data.
    """)


with st.expander("2. What data is expected in my CSV upload?"):
    st.write("""
    For the app to run predictions accurately, your CSV files should contain the following columns. Here’s a list of the required columns along with descriptions of the expected values:

    Employee Data File:
    """)
    
    st.table({
        "Column": [
            "Employee ID", "Full Name", "Gender", "Age", "Tenure",
            "Starting Salary", "Current Salary", "Promotion History", 
            "Last Performance Review Score", "Years of Experience",
            "Average Monthly Working Hours", "Promotion History", "Months in Role", "Department", 
            "Role", "Location", "Contract"
        ],
        "Expected Format": [
            "Integer", "String", "Categorical: Male/Female", "Integer", "Integer",
            "Float", "Float", "Integer", "Float", "Integer",
            "Float", "Integer", "Integer", "Categorical: e.g. Sales, IT, HR",
            "Categorical: e.g., Manager, Analyst, Engineer", 
            "Categorical: Remote/Office-based", "Categorical: Full-Time/Part-Time"
        ],
        "Description": [
            "Unique identifier for each employee.",
            "Full name of the employee.",
            "Gender of the employee.",
            "Age of the employee in years.",
            "Total number of years the employee has been with the company.",
            "Employee’s starting salary in the company.",
            "Current salary of the employee.",
            "Number of times the employee has been promoted during their tenure.",
            "The score from the last performance review between 1 and 5.",
            "Total number of years of relevant work experience.",
            "Average hours worked per month.",
            "Number of times the employee was promoted at the company",
            "Number of months the employee has spent in their current role.",
            "Department where the employee currently works.",
            "The specific role or title held by the employee.",
            "Location or type of working arrangement.",
            "Type of employment contract."
        ]
    })

    st.write("Benefits Data File:")

    st.table({
        "Column": [
            "Employee ID", "Category", "Enrollment Status"
        ],
        "Expected Format": [
            "Integer", "String", "Boolean"
        ],
        "Description": [
            "Unique identifier for each employee.",
            "Category of benefit, e.g. insurance, fitness, lunch.",
            "True if the employee is enrolled in the benefit. False if not."
        ]
    })

    st.write("Performance Review File:")

    st.table({
        "Column": [
            "Employee ID", 'Fiscal Quarter', 'Score', 'Performance Review Summary'
        ],
        "Expected Format": [
            "Integer", "String", "Integer", "String"
        ],
        "Description": [
            "Unique identifier for each employee.",
            "E.g. Q3 or Q4",
            "How the employee was assessed during the performance review, between 1 to 5.",
            "Short qualitative summary of the performance review."
        ]
    })

    st.write("Engagement Survey File:")

    st.table({
        "Column": [
            "Employee ID", 'Question', 'Score', 'Comment'
        ],
        "Expected Format": [
            "Integer", "String", "Integer", "String"
        ],
        "Description": [
            "Unique identifier for each employee.",
            "Name of the survey question, e.g. work-life balance.",
            "The employee's score, between 1 to 5.",
            "Short qualitative comment related to the question."
        ]
    })

    
with st.expander("3. What data is expected in my PDF upload?"):
    st.write("""
    The PDF upload feature allows you to provide additional documents that can enhance the model's insights. Accepted PDF files include:

    - **Industry Trends Reports**: Documents with information on hiring trends, compensation benchmarks, and other industry-specific insights.
    - **Company-Specific Benefits Documents**: Outlines of available company benefits, which help the model consider personalized factors in its retention recommendations.

    These documents will enrich the model’s knowledge base with qualitative information, making predictions more tailored to your organization’s context.

    To get a better sense of what information these PDFs should contain, try uploading one of our sample PDF files.
    """)


with st.expander("4. How does the tool process my data?"):
    st.write("""
    After data upload, the tool performs several steps:

    - **Data Cleaning & Preprocessing**: Cleans and processes each column to align with the tool’s machine learning model requirements.
    - **Feature Engineering**: Calculates additional insights, like `Salary Raise Per Year` and `Promotion Frequency`.
    - **Prediction**: Provides an "Attrition Probability" score for each employee.
    - **Retention Recommendation**: The Large Language Model writes personalized retention strategy based on the uploaded CSV and PDF files.
    """)


with st.expander("5. Can I download the results?"):
    st.write("""
    Yes, after generating a personalized retention recommendation, you can download it as PDF file.
    """)


with st.expander("6. Can I adjust the weights of the prediction model?"):
    st.write("""
    No, currently there is no option to adjust the weights of the attrition prediction model. The app is designed primarily for business users, not necessarily for technical users.
    """)
    

