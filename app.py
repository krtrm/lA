import os
import streamlit as st
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the app
st.set_page_config(
    page_title="Vaqeel - Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API endpoint (FastAPI backend)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 3px solid #3B82F6;
    }
    .source-title {
        font-weight: bold;
        color: #1E3A8A;
    }
    .source-url {
        color: #3B82F6;
        font-size: 0.9rem;
    }
    .response-container {
        background-color: #F0F9FF;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #BAE6FD;
    }
    .help-text {
        font-size: 0.85rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# App header
st.markdown("<h1 class='main-header'>Vaqeel.app</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your AI Legal Assistant for Indian Law</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1159/1159356.png", width=100)
    st.markdown("## About Vaqeel")
    st.markdown("""
    Vaqeel is an AI legal assistant trained on Indian law. 
    It can help with legal research, explain legal concepts, and 
    provide information on laws and cases.
    
    **Disclaimer:** Vaqeel provides information for educational purposes only. 
    It is not a substitute for professional legal advice.
    """)
    
    st.markdown("---")
    st.markdown("### Settings")
    web_search = st.checkbox("Enable web search", value=True, 
                          help="When enabled, Vaqeel will search the web for the most up-to-date information")
    
    st.markdown("---")
    st.markdown("### Sample Questions")
    sample_questions = [
        "What is the procedure to file an RTI in India?",
        "Explain the rights granted under Article 21 of the Indian Constitution",
        "What are the provisions for paternity leave in India?",
        "Explain the recent changes to the IT Act in India",
        "What is the process for filing a consumer complaint in India?"
    ]
    
    for q in sample_questions:
        if st.button(q):
            st.session_state.current_question = q

# Main chat interface
query = st.text_area("Ask any legal question about Indian law:", 
                     height=100, 
                     key="query_input",
                     value=st.session_state.get('current_question', ''))

col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button("Ask Vaqeel", type="primary")

if 'current_question' in st.session_state:
    del st.session_state.current_question

# Process the query
if submit_button and query:
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Display a spinner while waiting for the response
    with st.spinner("Vaqeel is researching your question..."):
        try:
            # Call the API
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query, "use_web": web_search},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                
                # Add response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": f"I encountered an error while researching your question. Please try again later."
                })
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"I encountered an error while researching your question. Please try again later."
            })

# Display chat history
if st.session_state.chat_history:
    st.markdown("### Conversation")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown("<div class='response-container'>", unsafe_allow_html=True)
            st.markdown(f"**Vaqeel:** {message['content']}")
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                st.markdown("---")
                st.markdown("**Sources:**")
                for source in message["sources"]:
                    st.markdown(f"""
                    <div class='source-box'>
                        <div class='source-title'>{source.get('title', 'Unknown Source')}</div>
                        <div class='source-url'>{source.get('source', '')}</div>
                        <div>{source.get('type', 'document').capitalize()}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Add a way to clear chat
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("<p class='help-text'>Vaqeel.app is powered by advanced AI technology and a comprehensive Indian legal database.</p>", unsafe_allow_html=True)
