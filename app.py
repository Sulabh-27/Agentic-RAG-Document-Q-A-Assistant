import os
import streamlit as st
import tempfile
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyPDFDirectoryLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize session state
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "agent_params" not in st.session_state:
    st.session_state.agent_params = None
if "reuse_query" not in st.session_state:
    st.session_state.reuse_query = None

# Sidebar settings
model_options = ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]
model_option = st.sidebar.selectbox("Select Model", model_options)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200)
k_docs = st.sidebar.slider("Number of Documents to Retrieve", 1, 10, 4)

with st.sidebar.expander("Advanced Settings"):
    enable_debug = st.checkbox("Enable Debug Mode", value=False)
    reasoning_depth = st.select_slider(
        "Reasoning Depth",
        options=["Basic", "Standard", "Deep"],
        value="Standard"
    )
    max_tokens = st.number_input("Max Tokens", min_value=256, max_value=8192, value=1024)

# Helper functions
def check_api_keys():
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY. Please add it to your .env file.")
        return False
    return True

def init_vector_store(documents=None):
    if st.session_state.vector_store and documents is None:
        return st.session_state.vector_store
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if documents:
            vector_store = FAISS.from_documents(documents, embeddings)
            st.session_state.vector_store = vector_store
            return vector_store
        return None
    except Exception as e:
        st.error(f"Error initializing FAISS vector store: {str(e)}")
        return None

def process_document(file_or_docs, source_name="unknown"):
    if not isinstance(file_or_docs, list):  # Uploaded file
        # Check file size (Streamlit limit is 200MB)
        file_size_mb = len(file_or_docs.getvalue()) / (1024 * 1024)  # Convert bytes to MB
        if file_size_mb > 200:
            raise ValueError(f"File {source_name} exceeds 200MB limit ({file_size_mb:.2f}MB)")

    file_id = str(uuid.uuid4())
    if isinstance(file_or_docs, list):  # Directory-loaded documents
        documents = file_or_docs
        st.write(f"[DEBUG] Loaded {len(documents)} documents from directory")
    else:  # Uploaded file
        suffix = os.path.splitext(file_or_docs.name)[1].lower()
        source_name = file_or_docs.name
        st.write(f"[DEBUG] Processing file: {source_name}, format: {suffix}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(file_or_docs.read())
            temp_path = temp.name
        try:
            st.write(f"[DEBUG] Loading document with appropriate loader")
            if suffix == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif suffix == ".docx":
                loader = Docx2txtLoader(temp_path)
            elif suffix == ".txt":
                loader = TextLoader(temp_path)
            elif suffix == ".csv":
                loader = CSVLoader(temp_path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            documents = loader.load()
            st.write(f"[DEBUG] Loaded {len(documents)} document pages")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                st.write(f"[DEBUG] Cleaned up temporary file: {temp_path}")

    st.write(f"[DEBUG] Adding metadata to documents")
    for doc in documents:
        doc.metadata["source"] = source_name
        doc.metadata["file_id"] = file_id

    st.write(f"[DEBUG] Splitting documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"[DEBUG] Split into {len(chunks)} chunks")

    st.write(f"[DEBUG] Adding chunks to FAISS vector store")
    try:
        for i, chunk in enumerate(chunks):
            if not chunk.page_content:
                st.write(f"[DEBUG] Warning: Chunk {i} is empty")
            else:
                st.write(f"[DEBUG] Chunk {i} length: {len(chunk.page_content)} characters")
        vector_store = init_vector_store(chunks)
        if not vector_store:
            raise ValueError("Failed to initialize FAISS vector store")
        st.write(f"[DEBUG] Successfully added chunks to FAISS vector store")
    except Exception as e:
        st.write(f"[DEBUG] Failed to add chunks to FAISS vector store: {str(e)}")
        try:
            st.write(f"[DEBUG] Attempting to embed a single chunk for debugging")
            test_chunk = chunks[0]
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embedding = embeddings.embed_documents([test_chunk.page_content])[0]
            st.write(f"[DEBUG] Successfully embedded a single chunk (length: {len(embedding)})")
        except Exception as embed_e:
            st.write(f"[DEBUG] Embedding failed: {str(embed_e)}")
        raise e

    return len(chunks), file_id

def clear_vector_store(file_id=None):
    try:
        if file_id:
            st.session_state.processed_files = [f for f in st.session_state.processed_files if f['file_id'] != file_id]
            if st.session_state.processed_files:
                # Rebuild FAISS index with remaining documents
                all_chunks = []
                for file in st.session_state.processed_files:
                    # Note: This requires reloading documents; in practice, you may want to store chunks
                    st.warning("Rebuilding FAISS index not fully implemented; clear all documents instead.")
                    st.session_state.vector_store = None
                    return
        st.session_state.vector_store = None
    except Exception as e:
        st.error(f"Error clearing FAISS vector store: {str(e)}")

def create_rag_tools():
    if not check_api_keys():
        return None
    try:
        llm = ChatGroq(
            model_name=model_option,
            groq_api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        vector_store = st.session_state.vector_store
        if not vector_store:
            raise ValueError("FAISS vector store not initialized")
        retriever = vector_store.as_retriever(search_kwargs={"k": k_docs})

        # Detailed answer tool
        detailed_template = """
        # Objective
        Answer the user's question based ONLY on the provided context from their documents.
        
        # Reasoning Process
        1. First, analyze the user's question: {question}
        2. Identify the key information needed to answer this question
        3. Carefully review the retrieved context
        4. Consider what the context says and doesn't say about the question
        
        # Retrieved Context
        {context}
        
        # Instructions
        - If the answer is clearly found in the context, provide it with specific references
        - If the context contains partial information, provide what's available and acknowledge limitations
        - If the context doesn't contain relevant information, state "I don't have enough information in the documents to answer this question"
        - DO NOT use information outside the provided context
        - Cite specific sources from the documents when possible
        """
        detailed_prompt = PromptTemplate(template=detailed_template, input_variables=["context", "question"])
        detailed_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": detailed_prompt},
            return_source_documents=True,
        )

        # Summarization tool
        summary_template = """
        Summarize the key points from the following document chunks in a concise paragraph:
        {context}
        """
        summary_prompt = PromptTemplate(template=summary_template, input_variables=["context"])
        summary_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": summary_prompt},
            return_source_documents=True,
        )

        def detailed_tool(query: str):
            result = detailed_qa({"query": query})
            response = result["result"]
            if "source_documents" in result and result["source_documents"]:
                sources = [f"{i+1}. {doc.metadata.get('source', 'Unknown')}" for i, doc in enumerate(result["source_documents"])]
                response += "\n\n**Sources:**\n" + "\n".join(sources)
            return response

        def summary_tool(query: str):
            result = summary_qa({"query": query})
            response = result["result"]
            if "source_documents" in result and result["source_documents"]:
                sources = [f"{i+1}. {doc.metadata.get('source', 'Unknown')}" for i, doc in enumerate(result["source_documents"])]
                response += "\n\n**Sources:**\n" + "\n".join(sources)
            return response

        return [
            Tool(
                name="Detailed Document Analysis",
                func=detailed_tool,
                description="Provides detailed answers with document references."
            ),
            Tool(
                name="Document Summarization",
                func=summary_tool,
                description="Summarizes key points from documents."
            )
        ]
    except Exception as e:
        st.error(f"Error creating RAG tools: {str(e)}")
        return None

def create_agent():
    if not check_api_keys():
        return None
    try:
        tools = create_rag_tools()
        if not tools:
            raise ValueError("RAG tools initialization failed")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        llm = ChatGroq(
            model_name=model_option,
            groq_api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        system_message = f"""
        You are an advanced document analysis assistant.
        Follow this reasoning process for each query:
        1. UNDERSTAND: Understand what the user is asking
        2. PLAN: Determine what information you need from the documents
        3. RETRIEVE: Use your tools to get relevant information
        4. ANALYZE: Examine the retrieved information
        5. SYNTHESIZE: Combine information if needed
        6. REASON: Draw logical conclusions
        7. RESPOND: Provide a clear, concise answer with source references
        
        Reasoning depth: {reasoning_depth}
        
        Guidelines:
        - Only use information from the user's documents
        - Cite specific sources
        - Acknowledge limitations
        - Be truthful and accurate
        """
        agent_chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=enable_debug,
            memory=memory,
            agent_kwargs={"system_message": system_message}
        )
        return agent_chain
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

def get_agent():
    params = (model_option, temperature, max_tokens, reasoning_depth, enable_debug)
    if "agent" not in st.session_state or st.session_state.agent_params != params:
        with st.spinner("Initializing agent..."):
            st.session_state.agent = create_agent()
            st.session_state.agent_params = params
    return st.session_state.agent

def display_reasoning(query):
    st.write("🧠 **Reasoning Process**")
    reasoning_steps = [
        "**Understanding Query**: Analyzing what information is being requested",
        "**Planning Retrieval**: Determining key terms and concepts to search for",
        "**Analyzing Documents**: Examining retrieved information for relevance",
        "**Synthesizing Answer**: Combining information from multiple sources"
    ]
    with st.expander("View Reasoning Steps", expanded=False):
        for step in reasoning_steps:
            st.write(step)
            st.progress(100)

# Streamlit UI
st.title("Document Q&A Assistant")

tab1, tab2, tab3 = st.tabs(["Manage Documents", "Chat with Documents", "Query History"])

with tab1:
    st.header("Manage Documents")
    st.write("Upload files or load PDFs from the 'documents' directory.")

    if not check_api_keys():
        st.warning("Please add GROQ_API_KEY to your .env file")

    # File upload
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT, CSV)",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )
    if uploaded_files and st.button("Process Uploaded Files"):
        with st.spinner("Processing files..."):
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                if file.name in [f["name"] for f in st.session_state.processed_files]:
                    st.warning(f"File {file.name} already processed. Skipping.")
                    continue
                try:
                    num_chunks, file_id = process_document(file)
                    st.session_state.processed_files.append({
                        "name": file.name,
                        "chunks": num_chunks,
                        "file_id": file_id,
                        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success(f"Processed {file.name} into {num_chunks} chunks")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {repr(e)}")
                progress_bar.progress((i + 1) / len(uploaded_files))

    # Directory loading
    st.subheader("Load PDFs from Directory")
    if st.button("Load PDFs from 'documents'"):
        if not os.path.exists("documents"):
            st.error("The 'documents' directory does not exist")
        else:
            try:
                loader = PyPDFDirectoryLoader("documents")
                documents = loader.load()
                if not documents:
                    st.warning("No PDFs found in 'documents' directory")
                else:
                    with st.spinner("Processing directory PDFs..."):
                        if "directory_pdfs" in [f["name"] for f in st.session_state.processed_files]:
                            st.warning("Directory PDFs already processed. Skipping.")
                        else:
                            num_chunks, file_id = process_document(documents, "directory_pdfs")
                            st.session_state.processed_files.append({
                                "name": "directory_pdfs",
                                "chunks": num_chunks,
                                "file_id": file_id,
                                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            st.success(f"Processed {len(documents)} PDFs into {num_chunks} chunks")
            except Exception as e:
                st.error(f"Error processing directory: {str(e)}")

    # Processed files
    if st.session_state.processed_files:
        st.subheader("Processed Documents")
        for file in st.session_state.processed_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {file['name']} - {file['chunks']} chunks (Processed: {file['processed_at']})")
            with col2:
                if st.button("Delete", key=f"delete_{file['file_id']}"):
                    clear_vector_store(file['file_id'])
                    st.rerun()
        if st.button("Clear All Documents", type="primary"):
            if st.checkbox("Confirm clearing all documents"):
                clear_vector_store()
                st.session_state.processed_files = []
                st.rerun()

with tab2:
    st.header("Ask Questions or Summarize Documents")
    if not st.session_state.processed_files:
        st.warning("Please process documents first")
    else:
        agent = get_agent()
        if not agent:
            st.error("Failed to initialize agent")
        else:
            query_type = st.selectbox("Query Type", ["Detailed Answer", "Summary"])
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            prompt = st.chat_input("Enter your question or request")
            if st.session_state.reuse_query:
                prompt = st.session_state.reuse_query
                st.session_state.reuse_query = None
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    if enable_debug:
                        display_reasoning(prompt)
                    with st.spinner("Processing query..."):
                        try:
                            tool_instruction = "Detailed Document Analysis" if query_type == "Detailed Answer" else "Document Summarization"
                            response = agent.run(f"Use {tool_instruction} tool: {prompt}")
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.session_state.query_history.append({
                                "query": prompt,
                                "response": response,
                                "type": query_type,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

with tab3:
    st.header("Query History")
    if not st.session_state.query_history:
        st.info("No queries yet")
    else:
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.rerun()
        for i, item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"{item['type']}: {item['query'][:50]}{'...' if len(item['query']) > 50 else ''}", expanded=(i == 0)):
                st.write(f"**Type:** {item['type']}")
                st.write(f"**Timestamp:** {item['timestamp']}")
                st.write(f"**Query:** {item['query']}")
                st.write(f"**Response:** {item['response']}")
                if st.button(f"Ask Again", key=f"reuse_{i}"):
                    st.session_state.reuse_query = item["query"]
                    st.rerun()