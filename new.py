import os
import gc
import uuid
import time
import tempfile
import requests
from urllib.parse import urlparse
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import streamlit as st
from IPython.display import Markdown, display

from llama_index.core.settings import Settings
from llama_index.llms.cerebras import Cerebras
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.experimental.query_engine import PandasQueryEngine, PandasInstructionParser

# Configure page
st.set_page_config(
    page_title="Enhanced Data Analysis & RAG Chatbot",
    page_icon="🪅",
    layout="centered"
)

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.current_engine = None
    st.session_state.data_type = None

session_id = st.session_state.id

@st.cache_resource
def load_llm():
    """Load Cerebras LLM"""
    llm = Cerebras(
        model="gpt-oss-120b", 
        api_key=os.getenv("CEREBRAS_API_KEY")
    )
    return llm

def reset_chat():
    """Reset chat history and context"""
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.current_engine = None
    st.session_state.data_type = None
    gc.collect()

def display_file_info(file, file_type="PDF"):
    """Display file information"""
    file_details = {
        "Filename": file.name, 
        "FileSize": f"{len(file.getvalue()) / 1024:.2f} KB",
        "Type": file_type
    }
    st.json(file_details)

def download_from_url(url, temp_dir):
    """Download content from URL and save to temp directory"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Get filename from URL or use default
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "downloaded_content.txt"
        
        file_path = os.path.join(temp_dir, filename)
        
        # Save content based on type
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' in content_type:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            # For text content, save as .txt
            if not filename.endswith('.txt'):
                filename = filename + '.txt'
                file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        return file_path, filename
    except Exception as e:
        st.error(f"Error downloading from URL: {str(e)}")
        return None, None

def create_rag_engine(docs):
    """Create RAG query engine with Cerebras LLM and HuggingFace embeddings"""
    llm = load_llm()
    
    try:
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.embed_model = embed_model
        Settings.llm = llm
    except Exception as e:
        st.error(f"Failed to initialize HuggingFace embeddings: {str(e)}")
        st.stop()
    
    # Create index and query engine
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    query_engine = index.as_query_engine(streaming=True)
    
    # Custom QA prompt
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above, I want you to think step by step to answer the query in a crisp manner. "
        "If you don't know the answer, say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )
    
    return query_engine

def create_pandas_engine(df):
    """Create pandas query engine for CSV/Excel analysis"""
    instruction_str = (
        "1. Convert the query to executable Python code using Pandas.\n"
        "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
        "3. The code should represent a solution to the query.\n"
        "4. PRINT ONLY THE EXPRESSION.\n"
        "5. Do not quote the expression.\n"
    )
    
    pandas_prompt_str = (
        "You are working with a pandas dataframe in Python.\n"
        "The name of the dataframe is `df`.\n"
        "This is the result of `print(df.head())`:\n"
        "{df_str}\n\n"
        "Follow these instructions:\n"
        "{instruction_str}\n"
        "Query: {query_str}\n\n"
        "Expression:"
    )
    
    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, 
        df_str=df.head(5)
    )
    pandas_output_parser = PandasInstructionParser(df)
    
    # Set LLM for pandas engine
    Settings.llm = load_llm()
    
    query_engine = PandasQueryEngine(
        df=df, 
        verbose=True, 
        pandas_prompt=pandas_prompt, 
        pandas_output_parser=pandas_output_parser
    )
    
    return query_engine

# Sidebar
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFv7810BCN31iB_-zp1CxNi8AXHEXODHSKr7XbiFNZrutzDqzXcJWIVv6TsgUQfwPVaYI&usqp=CAU", width='stretch')

with st.sidebar:
    st.header("📂 Upload Your Data")
    
    # Tab selection for different data types
    data_tab = st.selectbox(
        "Choose data type:",
        ["📄 Documents (PDF)", "📊 Structured Data (CSV/Excel)", "🌐 Website URL"]
    )
    
    if data_tab == "📄 Documents (PDF)":
        uploaded_file = st.file_uploader("Choose your PDF file", type="pdf")
        
        if uploaded_file:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    file_key = f"{session_id}-{uploaded_file.name}"
                    st.write("📑 Indexing your document...")
                    
                    if file_key not in st.session_state.get('file_cache', {}):
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            required_exts=[".pdf"],
                            recursive=True
                        )
                        docs = loader.load_data()
                        query_engine = create_rag_engine(docs)
                        st.session_state.file_cache[file_key] = query_engine
                    else:
                        query_engine = st.session_state.file_cache[file_key]
                    
                    st.session_state.current_engine = query_engine
                    st.session_state.data_type = "RAG"
                    st.success("✅ Ready to Chat!")
                    display_file_info(uploaded_file, "PDF")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    
    elif data_tab == "📊 Structured Data (CSV/Excel)":
        uploaded_file = st.file_uploader(
            "Choose your data file", 
            type=["csv", "xlsx", "xls"]
        )
        
        if uploaded_file:
            try:
                # Load data based on file type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.write("📊 Data Preview:")
                st.dataframe(df.head())
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Create pandas query engine
                query_engine = create_pandas_engine(df)
                st.session_state.current_engine = query_engine
                st.session_state.data_type = "PANDAS"
                st.session_state.df = df
                
                st.success("✅ Ready for Data Analysis!")
                display_file_info(uploaded_file, f"Data File ({uploaded_file.name.split('.')[-1].upper()})")
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    elif data_tab == "🌐 Website URL":
        url_input = st.text_input("Enter website URL:")
        if url_input and st.button("📥 Load Website"):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    st.write("🌐 Downloading content from URL...")
                    file_path, filename = download_from_url(url_input, temp_dir)
                    
                    if file_path:
                        file_key = f"{session_id}-{filename}"
                        
                        if file_key not in st.session_state.get('file_cache', {}):
                            loader = SimpleDirectoryReader(input_dir=temp_dir)
                            docs = loader.load_data()
                            query_engine = create_rag_engine(docs)
                            st.session_state.file_cache[file_key] = query_engine
                        else:
                            query_engine = st.session_state.file_cache[file_key]
                        
                        st.session_state.current_engine = query_engine
                        st.session_state.data_type = "RAG"
                        st.success("✅ Website content loaded successfully!")
                        st.json({"URL": url_input, "Filename": filename})
                        
            except Exception as e:
                st.error(f"Error loading website: {e}")

# Main chat interface
col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("<h3 style='color: #fc7303;'>🪅 RAG using Cerebras : GPT-OSS-120B </h3>", unsafe_allow_html=True)

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Verify API keys
api_key = os.getenv("CEREBRAS_API_KEY")
if not api_key:
    st.error("CEREBRAS_API_KEY not found in environment variables. Please add it to your .env file.")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "unsafe_allow_html=True" in message.get("content", ""):
            # For messages with HTML content
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your data..."):
    if not st.session_state.current_engine:
        st.warning("⚠️ Please upload a file first!")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        
        try:
            if st.session_state.data_type == "RAG":
                # RAG streaming response
                streaming_response = st.session_state.current_engine.query(prompt)
                response_chunks = []
                for chunk in streaming_response.response_gen:
                    if "<think>" in chunk or "</think>" in chunk:
                        continue
                    response_chunks.append(chunk)
                    full_response = "".join(response_chunks)
                    message_placeholder.markdown(full_response + " <img src='https://media.tenor.com/HiVVJv-skJcAAAAM/pac-man.gif' width='22' style='vertical-align: middle;'/>",unsafe_allow_html=True)
            
            elif st.session_state.data_type == "PANDAS":
                # Pandas response
                response = st.session_state.current_engine.query(prompt)
                full_response = str(response)
                message_placeholder.markdown(full_response)
            
            # Add response time
            end_time = time.time()
            response_time = end_time - start_time
            
            response_time_html = f"""
            <div style='margin-top: 10px; color: #28fa07; font-size: 0.9em;'>
                ⚡Response time: {response_time:.2f} seconds
            </div>
            """
            final_response = f"{full_response}{response_time_html}"
            message_placeholder.markdown(final_response, unsafe_allow_html=True)
            
            # Store in chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_response
            })
            
        except Exception as e:
            end_time = time.time()
            error_time = end_time - start_time
            error_msg = f"❌ Error generating response after {error_time:.2f} seconds: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_msg
            })

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Built with ❤️ using Streamlit • Powered by Cerebras & HuggingFace
    </div>
    """, 
    unsafe_allow_html=True
)
