import streamlit as st
from utils import get_chat_engine, get_query_engine, get_employee_snapshot
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import PromptTemplate


def main():
    st.title("Retain AI: Chat")
    
    
    if "employee data_df" not in st.session_state and "employee_mappings" not in st.session_state:
        st.warning("Start with Data Upload tab to upload your files or use sample dataset")
        return
    
    # in case when user switched sample data toggle multiple times and did not load the data
    elif st.session_state['demo_mode']==False and "employee_mappings" not in st.session_state:
        st.warning("Start with Data Upload tab to upload your files or use sample dataset")
        return
        
    
    # Get snapshots of all employees for passing to LLM context
    full_snapshot = "\n".join(
        get_employee_snapshot(st.session_state["employee data_df"].iloc[[index]]) 
        for index, _ in st.session_state["employee data_df"].iloc[:10].iterrows()
    )
    
    if "messages" not in st.session_state.keys():  # Initialize the chat messages history
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me a question your team",
            }
        ]
    
    #chat_engine = get_query_engine()
    chat_engine = get_chat_engine(context="")
    
    
    if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
        st.session_state.chat_engine = chat_engine
    
    if prompt := st.chat_input(
        "Ask a question"
    ):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    for message in st.session_state.messages:  # Write message history to UI
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # messages = [
    #     ChatMessage(
    #         role=MessageRole.SYSTEM, content=("You are a helpful assistant.")
    #     ),
    #     ChatMessage(
    #         role=MessageRole.USER,
    #         content=(prompt),
    #     ),
    # ]
    
    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )
    qa_template = PromptTemplate(template)
    
    messages = qa_template.format_messages(context_str=full_snapshot, query_str=prompt)
    
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = st.session_state.chat_engine.chat(messages)
    
            placeholder = st.empty()
            full_response = ""
            for chunk in response.message.content:
                full_response += chunk
                placeholder.markdown(full_response)
            
            message = {"role": "assistant", "content": full_response}
            # Add response to message history
            st.session_state.messages.append(message)

main()
