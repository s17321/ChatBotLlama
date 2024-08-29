import streamlit as st
import requests

# App title
st.set_page_config(page_title="🦙💬 University Learning Assistant Chatbot")

# FastAPI endpoint
API_URL = "http://localhost:8000/llm-input"

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 University Learning Assistant Chatbot')
    st.write('This chatbot is designed to assist students with their university courses. It can answer questions about course materials, provide explanations, and help with understanding various topics related to their studies.')
  
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response using FastAPI endpoint
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    response = requests.post(API_URL, params = {"user_input": prompt_input})

    if response.status_code == 200:
        return response.json().get("completion", "")
    else:
        return "Error: Unable to get response from the API"

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
