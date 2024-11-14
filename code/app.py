import streamlit as st

# Define the pages for the application with respective titles and icons
create_page = st.Page("data_upload.py", title="Data Upload", icon=":material/add_circle:")  # Page for uploading CSV and PDF files
delete_page = st.Page("dashboard.py", title="Dashboard", icon=":material/analytics:")       # Dashboard page to display attrition predictions
faq = st.Page("faq.py", title="FAQ", icon=":material/quiz:")                                # FAQ page with information about data requirements and app usage
chat = st.Page("chat.py", title="Chat", icon=":material/chat:")                             # Chat page for querying employee insights

# Set up the app navigation with the defined pages
pg = st.navigation([create_page, delete_page, chat, faq])

# Configure main page title and icon for the app
st.set_page_config(page_title="RetainAI", page_icon="🤖")

# Run the navigation to render the selected page content
pg.run()
