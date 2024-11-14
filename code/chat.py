import streamlit as st
from utils import get_chat_engine, get_query_engine, get_employee_snapshot, rename_and_filter_columns
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import PromptTemplate

# Main function for the chat interface
def main():
    # Set up the title for the chat page
    st.title("Retain AI: Chat")
    
    # Check if essential data is loaded; if not, prompt user to start with data upload
    if "employee data_df" not in st.session_state and "employee_mappings" not in st.session_state:
        st.warning("Start with Data Upload tab to upload your files or use sample dataset")
        return
    
    # Ensure data is loaded properly, handling cases where demo mode is toggled without loading data
    elif st.session_state['demo_mode'] == False and "employee_mappings" not in st.session_state:
        st.warning("Start with Data Upload tab to upload your files or use sample dataset")
        return

    # Load employee data, applying renaming and filtering based on mappings if not in demo mode
    df = (
        st.session_state["employee data_df"]
        if st.session_state["demo_mode"]
        else rename_and_filter_columns(st.session_state["employee data_df"], st.session_state["employee_mappings"])
    )

    # Generate a full snapshot of all employees to provide necessary context for the LLM model
    full_snapshot = "\n".join(
        get_employee_snapshot(df.iloc[[index]]) 
        for index, _ in df.iloc[:10].iterrows()
    )

    # Initialize chat messages history if it doesn't already exist
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me a question about your team",
            }
        ]

    # Initialize the chat engine instance
    chat_engine = get_chat_engine()
    
    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = chat_engine

    # Display input prompt to allow user to ask a question and add it to message history
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display chat history on the UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Define prompt template with context and user query for the LLM to generate responses
    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )
    qa_template = PromptTemplate(template)
    
    # Format the messages with context (employee snapshot) and user query
    messages = qa_template.format_messages(context_str=full_snapshot, query_str=prompt)
    
    # If last message is from the user, generate a response from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = st.session_state.chat_engine.chat(messages)

            # Stream the assistant's response to the UI as it is generated
            placeholder = st.empty()
            full_response = ""
            for chunk in response.message.content:
                full_response += chunk
                placeholder.markdown(full_response)
            
            # Add the assistant's response to message history
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

# Run the main function to launch the chat interface
main()
