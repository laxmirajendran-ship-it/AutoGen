import streamlit as st
import asyncio
from autogen_workflow import run_autogen_workflow
from pathlib import Path
import json
st.set_page_config(layout="wide")

st.title("🤖 AutoGen Multi-Agent Test Case Generation")

# Sidebar for configuration
st.sidebar.title("Configuration")

config_path = Path(__file__).parent / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)


if(config.get("use_ollama") == True):
    st.sidebar.text_input("Ollama Model Name", config.get("ollama_model_name"),disabled=True)
else:
    st.sidebar.text_input("OpenAI Model Name", config.get("openai_model_name"),disabled=True)

temperature = st.sidebar.text_input("Temperature", config.get("temperature", 0.0),disabled=True)

# Main app layout
if "messages" not in st.session_state:
    st.session_state.messages = []

async def stream_workflow(prompt: str):
    """
    Streams the workflow output to the Streamlit chat interface.
    """
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        current_source = ""
        
        # Asynchronously iterate through the generator from run_autogen_workflow
        async for message in run_autogen_workflow(prompt):
            if message["content"]:
                # Check if the speaker has changed to add a header
                if current_source != message["source"]:
                    current_source = message["source"]
                    if full_response: # Add space between different agent messages
                        full_response += "\n\n"
                    full_response += f"**{current_source}:**\n"
                
                full_response += message["content"]
                message_placeholder.markdown(full_response + "▌")
                
        message_placeholder.markdown(full_response)
        
    # Append the full conversation to session state history
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# Chat input for the user
if prompt := st.chat_input("Enter your requirements here (e.g., 'password reset for online banking')"):
    workflow_successful = False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🤖 Agents are collaborating..."):
        try:
            loop.run_until_complete(stream_workflow(prompt))
            workflow_successful = True
        except Exception as e:
            st.error(f"An error occurred during the workflow. See details below.")
            st.exception(e) # This will print the full traceback for debugging

    if workflow_successful:
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])