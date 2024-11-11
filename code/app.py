import streamlit as st


create_page = st.Page("data_upload.py", title="Data Upload", icon=":material/add_circle:")
delete_page = st.Page("dashboard.py", title="Dashboard", icon=":material/analytics:")
faq = st.Page("faq.py", title="FAQ", icon=":material/quiz:")

pg = st.navigation([create_page, delete_page, faq])
st.set_page_config(page_title="RetainAI", page_icon=":material/edit:")
pg.run()
