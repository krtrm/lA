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
    page_title="LegalEase - Legal AI Assistant",
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
    .keyword-box {
        background-color: #F0FDF4;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #22C55E;
    }
    .term {
        font-weight: bold;
        color: #166534;
    }
    .definition {
        color: #374151;
    }
    .outline-container {
        background-color: #FEFCE8;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #FEF08A;
    }
    .citation-container {
        background-color: #FDF2F8;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #FBCFE8;
    }
    .citation-valid {
        background-color: #ECFDF5;
        border-left: 3px solid #10B981;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    .citation-invalid {
        background-color: #FEF2F2;
        border-left: 3px solid #EF4444;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    .tabs-font {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Chat"

# Add this callback function to handle query submission and reset
def submit_query():
    if st.session_state.query_input:
        st.session_state.submitted_query = st.session_state.query_input
        # Don't try to reset the input field directly here

# Modify the chat query handling function
def handle_chat_query():
    if st.session_state.get('submitted_query'):
        query = st.session_state.submitted_query
        # Clear the submitted query so we don't process it again
        st.session_state.submitted_query = ""
        web_search = st.session_state.get('web_search', True)
        with st.spinner("LegalEase is researching your question..."):
            try:
                response = requests.post(
                    f"{API_URL}/query/stream",
                    json={"query": query, "use_web": web_search, "stream_thinking": True},
                    stream=True,
                    timeout=60
                )
                thinking_placeholder = st.empty()
                answer = ""
                sources = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    step = json.loads(line.decode('utf-8'))
                    t = step.get('type')
                    c = step.get('content', '')
                    if t in ['thinking','planning','tool_use','retrieval','generation']:
                        thinking_placeholder.text(f"{t.capitalize()}: {c}")
                    elif t == 'complete':
                        answer = c
                        # Handle cases where details might be None
                        details = step.get('details', {})
                        if details is not None:
                            sources = details.get('sources', []) or []
                        break
                    elif t == 'error':
                        thinking_placeholder.text(f"Error: {c}")
                        break
                
                # Add the response to chat history
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": query
                })
                
                if answer:
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                else:
                    st.error("No response received from the AI.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# App header
st.markdown("<h1 class='main-header'>LegalEase.app</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your AI Legal Assistant for Indian Law</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1159/1159356.png", width=100)
    st.markdown("## About LegalEase")
    st.markdown("""
    LegalEase is an AI legal assistant trained on Indian law. 
    It can help with legal research, explain legal concepts, and 
    provide information on laws and cases.
    
    **Disclaimer:** LegalEase provides information for educational purposes only. 
    It is not a substitute for professional legal advice.
    """)
    
    st.markdown("---")
    st.markdown("### Choose Function")
    
    function_options = [
        "Chat", 
        "Legal Keyword Extraction", 
        "Legal Argument Composer", 
        "Document Outline Generator", 
        "Citation Verifier"
    ]
    
    selected_function = st.radio(
        "Select a function", 
        function_options,
        key="function_selector",
        index=function_options.index(st.session_state.current_tab)
    )
    st.session_state.current_tab = selected_function
    
    if selected_function == "Chat":
        st.markdown("### Chat Settings")
        web_search = st.checkbox("Enable web search", value=True, 
                            help="When enabled, LegalEase will search the web for the most up-to-date information")
        
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

# Main content area based on selected function
if st.session_state.current_tab == "Chat":
    # Chat interface
    query = st.text_area("Ask any legal question about Indian law:", 
                        height=100, 
                        key="query_input",
                        value=st.session_state.get('current_question', ''))

    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Ask LegalEase", type="primary", on_click=submit_query)

    if 'current_question' in st.session_state:
        del st.session_state.current_question

    # Process the query - call handle_chat_query on each render to process any submitted queries
    if st.session_state.get('submitted_query'):
        handle_chat_query()

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown("<div class='response-container'>", unsafe_allow_html=True)
                st.markdown(f"**LegalEase:** {message['content']}")
                
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

# Update Keyword Extraction to streaming
elif st.session_state.current_tab == "Legal Keyword Extraction":
    st.markdown("### Legal Keyword Extraction")
    st.markdown("""
    Extract key legal terms and their definitions from your text. 
    Perfect for understanding complex legal documents or preparing study materials.
    """)
    
    legal_text = st.text_area("Paste legal text for keyword extraction:", 
                             height=200, 
                             key="legal_text_input")
    
    if st.button("Extract Keywords", type="primary") and legal_text:
        with st.spinner("Extracting legal keywords..."):
            try:
                response = requests.post(
                    f"{API_URL}/extract_keywords/stream",
                    json={"text": legal_text},
                    stream=True,
                    timeout=60
                )
                thinking_placeholder = st.empty()
                terms = {}
                for line in response.iter_lines():
                    if not line: continue
                    step = json.loads(line.decode('utf-8'))
                    if step.get('type') in ['thinking','retrieval','generation']:
                        thinking_placeholder.text(f"{step['type'].capitalize()}: {step['content']}")
                    elif step.get('type') == 'complete':
                        details = step.get('details', {})
                        terms = details.get('terms', {})
                        break
                    elif step.get('type') == 'error':
                        thinking_placeholder.text(f"Error: {step['content']}")
                        break
                # Display results
                if terms:
                    st.success(f"Found {len(terms)} legal terms!")
                    # Display terms in a nice format
                    st.markdown("### Extracted Legal Terms")
                    for term, definition in terms.items():
                        st.markdown(f"""
                        <div class='keyword-box'>
                            <div class='term'>{term}</div>
                            <div class='definition'>{definition}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a download button
                        terms_json = json.dumps(terms, indent=2)
                        st.download_button(
                            label="Download Terms as JSON",
                            data=terms_json,
                            file_name="legal_terms.json",
                            mime="application/json"
                        )
                else:
                    st.info("No specialized legal terms were found in the text.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Update Argument Composer to streaming
elif st.session_state.current_tab == "Legal Argument Composer":
    st.markdown("### Legal Argument Composer")
    st.markdown("""
    Generate structured legal arguments ready to paste into your documents.
    Provide a topic and key points to include, and LegalEase will compose a coherent legal argument.
    """)
    
    topic = st.text_input("Main legal topic or issue:", key="argument_topic")
    
    # Dynamically add points
    if 'argument_points' not in st.session_state:
        st.session_state.argument_points = ["", "", ""]
    
    st.markdown("#### Key points to include:")
    
    # Display existing points
    for i, point in enumerate(st.session_state.argument_points):
        st.session_state.argument_points[i] = st.text_input(f"Point {i+1}", value=point, key=f"point_{i}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Add Point") and len(st.session_state.argument_points) < 10:
            st.session_state.argument_points.append("")
            st.experimental_rerun()
    with col2:
        if st.button("Remove Point") and len(st.session_state.argument_points) > 1:
            st.session_state.argument_points.pop()
            st.experimental_rerun()
    
    if st.button("Generate Argument", type="primary") and topic:
        # Filter out empty points
        points = [p for p in st.session_state.argument_points if p.strip()]
        
        if not points:
            st.warning("Please add at least one key point for your argument.")
        else:
            with st.spinner("Composing legal argument..."):
                try:
                    response = requests.post(
                        f"{API_URL}/generate_argument/stream",
                        json={"topic": topic, "points": points},
                        stream=True,
                        timeout=90
                    )
                    thinking_placeholder = st.empty()
                    argument = ""
                    for line in response.iter_lines():
                        if not line: continue
                        step = json.loads(line.decode('utf-8'))
                        if step.get('type') in ['thinking','retrieval','generation']:
                            thinking_placeholder.text(f"{step['type'].capitalize()}: {step['content']}")
                        elif step.get('type') == 'complete':
                            argument = step.get('content', '')
                            details = step.get('details', {})
                            break
                        elif step.get('type') == 'error':
                            thinking_placeholder.text(f"Error: {step['content']}")
                            break
                    if argument:
                        st.markdown("### Generated Legal Argument")
                        st.markdown(argument)
                        st.download_button(label="Download as Markdown", data=argument, file_name="legal_argument.md", mime="text/markdown")
                        word_count = details.get('word_count', len(argument.split()))
                        st.info(f"Argument length: {word_count} words")
                    else:
                        st.info("Could not generate an argument with the provided information.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Update Outline Generator to streaming
elif st.session_state.current_tab == "Document Outline Generator":
    st.markdown("### Document Outline Generator")
    st.markdown("""
    Get a professional document structure for your legal documents. 
    Select the document type and enter your topic to receive a comprehensive outline.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        doc_type = st.selectbox(
            "Document Type", 
            ["Legal Brief", "Legal Memo", "Legal Opinion", "Contract", "Petition", 
             "Complaint", "Settlement Agreement", "Legal Notice", "Affidavit", "Will"]
        )
    with col2:
        outline_topic = st.text_input("Document Topic:", key="outline_topic")
    
    if st.button("Generate Outline", type="primary") and outline_topic and doc_type:
        with st.spinner("Generating document outline..."):
            try:
                response = requests.post(
                    f"{API_URL}/create_outline/stream",
                    json={"topic": outline_topic, "doc_type": doc_type},
                    stream=True,
                    timeout=60
                )
                thinking_placeholder = st.empty()
                outline_text = ""
                for line in response.iter_lines():
                    if not line: continue
                    step = json.loads(line.decode('utf-8'))
                    if step.get('type') in ['thinking','retrieval','generation']:
                        thinking_placeholder.text(f"{step['type'].capitalize()}: {step['content']}")
                    elif step.get('type') == 'complete':
                        outline_text = step.get('content', '')
                        details = step.get('details', {})
                        break
                    elif step.get('type') == 'error':
                        thinking_placeholder.text(f"Error: {step['content']}")
                        break
                if outline_text:
                    st.markdown("### Document Outline")
                    st.markdown(f"<div class='outline-container'>{outline_text}</div>", unsafe_allow_html=True)
                    st.download_button(label="Download Outline as Markdown", data=outline_text, file_name=f"{doc_type.lower().replace(' ','_')}_outline.md", mime="text/markdown")
                    st.info(f"Outline structure: {details.get('section_count',0)} main sections with {details.get('subsection_count',0)} subsections")
                else:
                    st.info("Could not generate an outline with the provided information.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Update Citation Verifier to streaming
elif st.session_state.current_tab == "Citation Verifier":
    st.markdown("### Citation Verifier")
    st.markdown("""
    Verify the accuracy of legal citations, get corrections for improper formats, 
    and check if citations refer to valid cases or statutes.
    """)
    
    citation = st.text_input("Enter legal citation to verify:", key="citation_input", 
                           placeholder="e.g., AIR 1950 SC 27 or (2019) 1 SCC 1")
    
    if st.button("Verify Citation", type="primary") and citation:
        with st.spinner("Verifying citation..."):
            try:
                response = requests.post(
                    f"{API_URL}/verify_citation/stream",
                    json={"citation": citation},
                    stream=True,
                    timeout=60
                )
                thinking_placeholder = st.empty()
                result_data = {}
                for line in response.iter_lines():
                    if not line: continue
                    step = json.loads(line.decode('utf-8'))
                    if step.get('type') in ['thinking','tool_use','retrieval','generation']:
                        thinking_placeholder.text(f"{step['type'].capitalize()}: {step['content']}")
                    elif step.get('type') == 'complete':
                        result_data = step.get('details', {})
                        break
                    elif step.get('type') == 'error':
                        thinking_placeholder.text(f"Error: {step['content']}")
                        break
                if result_data:
                    # Display verification result
                    original = result_data.get("original_citation", citation)
                    is_valid = result_data.get("is_valid", False)
                    corrected = result_data.get("corrected_citation", "")
                    summary = result_data.get("summary", "")
                    error_details = result_data.get("error_details", "")
                    
                    st.markdown("<div class='citation-container'>", unsafe_allow_html=True)
                    
                    # Original citation
                    st.markdown(f"**Original Citation:** {original}")
                    
                    # Valid or Invalid indicator
                    if is_valid:
                        st.markdown("""
                        <div class='citation-valid'>
                            <strong>✓ Valid Citation</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='citation-invalid'>
                            <strong>✗ Invalid Citation</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Corrected citation if available
                    if corrected and corrected != original:
                        st.markdown(f"**Corrected Citation:** {corrected}")
                    
                    # Summary of cited case/law
                    if summary:
                        st.markdown("**Summary:**")
                        st.markdown(summary)
                    
                    # Error details if invalid
                    if not is_valid and error_details:
                        st.markdown("**Issues:**")
                        st.markdown(error_details)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("No verification data received.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p class='help-text'>LegalEase.app is powered by advanced AI technology and a comprehensive Indian legal database.</p>", unsafe_allow_html=True)
