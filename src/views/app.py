import os
import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import UPLOAD_DIR, CHUNK_SIZE as DEFAULT_CHUNK_SIZE
from src.preprocessing_data import load_data, split_docs
from src.vectorstore import load_embeddings, load_vectorstore
from src.rag_pipeline import get_retriever, rag_answer
from src.model_loader import load_llm

from src.views.components import (
    render_sidebar,
    render_chat_interface,
    render_document_info
)
# Page configuration
st.set_page_config(
    page_title="Document RAG Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChat {
        padding: 20px;
    }
    .stAlert {
        padding: 10px;
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "embeddings" not in st.session_state:
        with st.spinner("Loading embeddings model..."):
            st.session_state.embeddings = load_embeddings()

def process_uploaded_file(uploaded_file, chunk_size):
    """Process the uploaded file and create vectorstore"""
    try:
        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Update file_path in config (temporary solution)
        from src import config
        config.file_path = file_path
        
        # Load and process the document
        with st.spinner("📖 Loading document..."):
            documents = load_data()
        
        with st.spinner("✂️ Splitting into chunks..."):
            chunks = split_docs(documents, chunk_size=chunk_size)
        
        with st.spinner("💾 Creating vector database..."):
            # Create vectorstore
            vectorstore = load_vectorstore(chunks=chunks)
            retriever = get_retriever(vectorstore)
        
        return vectorstore, retriever, len(chunks), file_path
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None, 0, None

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header"><h1>📚 Document Q&A System</h1><p>Upload a document and start asking questions!</p></div>', 
                unsafe_allow_html=True)
    
    # Render sidebar and get inputs
    uploaded_file, k_retrieval = render_sidebar()
    
    # Main chat area
    messages = render_chat_interface()
    
    # Handle file upload
    if uploaded_file is not None:
        # Check if it's a new file
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing your document..."):
                vectorstore, retriever, num_chunks, file_path = process_uploaded_file(
                    uploaded_file, chunk_size=DEFAULT_CHUNK_SIZE
                )
                
                if vectorstore and retriever:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = retriever
                    st.session_state.current_file = uploaded_file.name
                    
                    # Show success message and document info
                    st.success(f"✅ File '{uploaded_file.name}' processed successfully!")
                    render_document_info(uploaded_file.name, num_chunks)
                    
                    # Clear chat history for new document
                    st.session_state.messages = []
                    
                    # Add welcome message
                    welcome_msg = f"Hello! I'm ready to answer questions about '{uploaded_file.name}'. What would you like to know?"
                    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                    
                    # Rerun to show welcome message
                    st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Check if document is loaded
        if st.session_state.retriever is None:
            st.warning("⚠️ Please upload a document first!")
            st.stop()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get answer
                    response = rag_answer(
                        name=st.session_state.current_file,
                        query=prompt,
                        retriever=st.session_state.retriever
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to session state
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.current_file:
                welcome_msg = f"Chat cleared! Still ready to answer about '{st.session_state.current_file}'."
                st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
            st.rerun()

if __name__ == "__main__":
    main()
