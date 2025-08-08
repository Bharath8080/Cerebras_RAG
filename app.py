import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import base64
import gc
import random
import tempfile
import time
import uuid

from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.llms.cerebras import Cerebras
from llama_index.embeddings.cohere import CohereEmbedding


from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

import streamlit as st

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

@st.cache_resource
def load_llm():

    llm = Cerebras(model="gpt-oss-120b", 
                         api_key=os.getenv("CEREBRAS_API_KEY"))
    return llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    # Just show file info without preview
    file_details = {"Filename": file.name, "FileSize": f"{len(file.getvalue()) / 1024:.2f} KB"}
    st.json(file_details)

# Sidebar logo
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cerebras_logo.svg/2560px-Cerebras_logo.svg.png", use_container_width=True)
with st.sidebar:
    
    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            required_exts=[".pdf"],
                            recursive=True
                        )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    docs = loader.load_data()
                    llm = load_llm()
                    
                    # Get and verify Cohere API key
                    cohere_api_key = os.getenv("COHERE_API_KEY")
                    if not cohere_api_key:
                        st.error("COHERE_API_KEY not found in environment variables")
                        st.stop()
                        
                    try:
                        embed_model = CohereEmbedding(
                            cohere_api_key=cohere_api_key,
                            model_name="embed-english-v3.0",
                            input_type="search_document"
                        )
                        Settings.embed_model = embed_model
                    except Exception as e:
                        st.error(f"Failed to initialize Cohere embeddings: {str(e)}")
                        st.stop()
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, in case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    # Removed the original header
    st.markdown("<h2 style='color: #1407fa;'> RAG using Cerebras : GPT-OSS-120B </h2>", unsafe_allow_html=True)



with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Load the Cerebras API key from environment variables
api_key = os.getenv("CEREBRAS_API_KEY")
if not api_key:
    st.error("CEREBRAS_API_KEY not found in environment variables. Please add it to your .env file.")
    st.stop()

# Configure the LLM with the API key
Settings.llm = load_llm()
Settings.llm.api_key = api_key

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Start timing the query
        import time
        start_time = time.time()
        
        try:
            streaming_response = query_engine.query(prompt)
            
            # Process the streaming response
            response_chunks = []
            for chunk in streaming_response.response_gen:
                if "<think>" in chunk or "</think>" in chunk:
                    continue
                response_chunks.append(chunk)
                full_response = "".join(response_chunks)
                message_placeholder.markdown(full_response + "▌")
            
            # Calculate and display response time with colored text
            end_time = time.time()
            response_time = end_time - start_time
            
            # Add response time with colored text
            response_time_html = f"""
            <div style='margin-top: 10px; color: #28fa07; font-size: 0.9em;'>
                Response time: {response_time:.2f} seconds
            </div>
            """
            full_response = f"{full_response}{response_time_html}"
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
            
        except Exception as e:
            end_time = time.time()
            error_time = end_time - start_time
            error_msg = f"Error generating response after {error_time:.2f} seconds: {str(e)}"
            message_placeholder.error(error_msg)
            full_response = error_msg
            
        # Store the final response in the chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
