# src/views/components.py
import streamlit as st
import os
from pathlib import Path

def render_sidebar():
    """Render the sidebar with file upload and configuration options"""
    with st.sidebar:
        st.title("📁 Document Upload")
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt', 'md', 'pdf'],
            help="Upload a text or markdown file to chat with"
        )
        
        st.markdown("---")
        
        # Configuration options
        st.subheader("⚙️ Settings")
        
        # Number of retrieved documents
        k_retrieval = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="How many relevant documents to retrieve for answering"
        )
        
        # Model selection (if you want to add this feature)
        st.markdown("---")
        st.subheader("🤖 Model Info")
        st.info(f"Using: mistral:latest")
        
        return uploaded_file, k_retrieval

def render_chat_interface():
    """Render the main chat interface"""
    st.title("💬 Chat with Your Document")
    st.markdown("---")
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    return st.session_state.messages

def render_document_info(uploaded_file_name, num_chunks):
    """Render information about the uploaded document"""
    with st.expander("📄 Document Info", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filename", uploaded_file_name)
        with col2:
            st.metric("Chunks created", num_chunks)