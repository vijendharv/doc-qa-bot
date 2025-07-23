# Document Q&A Bot using LangChain and Streamlit
# This application allows users to upload PDF documents and ask questions about their content
# using OpenAI's GPT models and vector similarity search

import streamlit as st
import os
from typing import List
# Updated imports for newer LangChain versions (0.1.0+)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import tempfile

class DocumentQABot:
    """
    Main class that handles the document Q&A functionality.
    
    This class encapsulates all the LangChain components needed for:
    1. Loading and processing PDF documents
    2. Creating vector embeddings for similarity search
    3. Setting up the question-answering chain
    4. Handling user queries and returning answers with sources
    """
    
    def __init__(self):
        """
        Initialize the DocumentQABot class.
        
        Sets up the initial state with empty vectorstore and qa_chain.
        These will be populated when a document is uploaded and processed.
        """
        self.vectorstore = None  # Will hold the FAISS vector index
        self.qa_chain = None     # Will hold the RetrievalQA chain
        self.setup_components()  # Initialize LangChain components
    
    def setup_components(self):
        """
        Initialize non-OpenAI components that don't require API keys.
        
        OpenAI components (embeddings and LLM) will be initialized later
        when the API key is provided by the user.
        """
        # Initialize text splitter for breaking documents into chunks
        # This is crucial because LLMs have token limits and work better with smaller chunks
        # This component doesn't need an API key, so we can initialize it immediately
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Each chunk will be ~1000 characters
            chunk_overlap=200,  # 200 character overlap between chunks to preserve context
            length_function=len # Use character count (not token count) for simplicity
        )
        
        # OpenAI components will be initialized in setup_openai_components()
        # when the user provides their API key
        self.embeddings = None
        self.llm = None
    
    def setup_openai_components(self, api_key: str):
        """
        Initialize OpenAI components once the API key is provided.
        
        Args:
            api_key: The user's OpenAI API key
            
        This method is called only after the user enters their API key
        to avoid initialization errors with missing credentials.
        """
        # Set the API key in environment variables
        # This is how the OpenAI client reads the authentication
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize OpenAI embeddings model
        # This converts text into numerical vectors that can be compared for similarity
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key  # Explicitly pass the API key
        )
        
        # Initialize the language model for generating answers
        # Using GPT-3.5-turbo with low temperature for consistent, factual responses
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",     # Fast and cost-effective model
            temperature=0.1,           # Low temperature = more deterministic responses
            max_tokens=512,            # Limit response length to control costs
            openai_api_key=api_key     # Explicitly pass the API key
        )
    
    def load_and_process_pdf(self, uploaded_file) -> List[Document]:
        """
        Load a PDF file and convert it into processable document chunks.
        
        Args:
            uploaded_file: Streamlit uploaded file object containing the PDF
            
        Returns:
            List[Document]: List of document chunks ready for embedding
            
        Process:
        1. Save the uploaded file to a temporary location
        2. Use PyPDFLoader to extract text from the PDF
        3. Split the text into chunks using the text splitter
        4. Clean up temporary files
        """
        try:
            # Create a temporary file to save the uploaded PDF
            # This is necessary because PyPDFLoader expects a file path, not file content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())  # Write uploaded file content
                tmp_path = tmp_file.name                  # Get the temporary file path
            
            # Use LangChain's PyPDFLoader to extract text from the PDF
            # This handles PDF parsing complexities like fonts, encoding, and layout
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()  # Returns list of Document objects (one per page)
            
            # Clean up the temporary file to free disk space
            os.unlink(tmp_path)
            
            # Split the documents into smaller chunks for better retrieval
            # Large documents need to be chunked because:
            # 1. LLMs have context limits
            # 2. Smaller chunks provide more precise retrieval
            # 3. Better similarity matching with user queries
            chunks = self.text_splitter.split_documents(documents)
            
            return chunks
        
        except Exception as e:
            # Display error message to user if PDF processing fails
            st.error(f"Error processing PDF: {str(e)}")
            return []
    
    def create_vectorstore(self, chunks: List[Document]):
        """
        Create a FAISS vector store from document chunks.
        
        Args:
            chunks: List of document chunks to be embedded
            
        Process:
        1. Convert each chunk into a vector using OpenAI embeddings
        2. Create a FAISS index for fast similarity search
        3. Store the vectorstore for later querying
        
        Note: FAISS (Facebook AI Similarity Search) is a library for efficient
        similarity search of dense vectors. It creates an index that allows
        us to quickly find the most relevant document chunks for any query.
        """
        try:
            # Check if OpenAI components are initialized
            if not self.embeddings:
                st.error("OpenAI components not initialized. Please enter your API key first.")
                return
            
            print(len(chunks))
            for chunk in chunks:
                print(chunk)
                
            if chunks:
                # Create FAISS vectorstore from document chunks
                # This process:
                # 1. Sends each chunk to OpenAI's embedding API
                # 2. Gets back numerical vectors representing the semantic content
                # 3. Builds a FAISS index for fast similarity search
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                st.success(f"Successfully processed {len(chunks)} document chunks!")
            else:
                st.error("No valid chunks found in the document")
        except Exception as e:
            # Handle errors like API failures, network issues, or invalid embeddings
            st.error(f"Error creating vectorstore: {str(e)}")
    
    def setup_qa_chain(self):
        """
        Set up the question-answering chain using the created vectorstore.
        
        The QA chain combines:
        1. A retriever that finds relevant document chunks
        2. An LLM that generates answers based on retrieved context
        3. Source tracking to show which documents were used
        
        This creates a RAG (Retrieval-Augmented Generation) system where
        the model's responses are grounded in the uploaded document content.
        """
        # Check if all required components are initialized
        if not self.vectorstore:
            st.error("Please upload and process a document first.")
            return
            
        if not self.llm:
            st.error("OpenAI components not initialized. Please enter your API key first.")
            return
            
        # Create a retriever from the vectorstore
        # The retriever finds the most similar document chunks to a query
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",    # Use cosine similarity for matching
            search_kwargs={"k": 3}       # Return top 3 most relevant chunks
        )
        
        # Create the RetrievalQA chain
        # This combines retrieval + generation into a single chain:
        # Query -> Retrieve relevant docs -> Generate answer using docs as context
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,                        # Language model for generation
            chain_type="stuff",                  # "Stuff" all retrieved docs into prompt
            retriever=retriever,                 # Component that finds relevant docs
            return_source_documents=True,        # Include source docs in response
            verbose=True                         # Print chain execution details
        )
    
    def ask_question(self, question: str) -> dict:
        """
        Process a user question and return an answer with sources.
        
        Args:
            question: User's natural language question about the document
            
        Returns:
            dict: Contains 'result' (answer) and 'source_documents' (sources)
                  or 'error' if something went wrong
                  
        Process:
        1. Use the retriever to find relevant document chunks
        2. Pass the question + relevant chunks to the LLM
        3. Generate a natural language answer
        4. Return answer along with source documents for transparency
        """
        # Check if the QA chain has been set up
        if not self.qa_chain:
            return {"error": "Please upload a document first!"}
        
        try:
            # Execute the RetrievalQA chain
            # This internally:
            # 1. Converts question to embeddings
            # 2. Finds similar document chunks using FAISS
            # 3. Creates a prompt with question + retrieved context
            # 4. Sends to LLM for answer generation
            # 5. Returns structured response with sources
            response = self.qa_chain({"query": question})
            return response
        except Exception as e:
            # Handle API errors, timeout issues, or malformed responses
            return {"error": f"Error processing question: {str(e)}"}

def main():
    """
    Main Streamlit application function.
    
    Sets up the web interface with:
    1. File upload functionality
    2. Document processing controls
    3. Chat interface for questions
    4. Source document display
    """
    
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Document Q&A Bot",  # Browser tab title
        page_icon="📚",                 # Favicon
        layout="wide"                   # Use full browser width
    )
    
    # Display main title and description
    st.title("📚 Document Q&A Bot")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Initialize the DocumentQABot in Streamlit session state
    # Session state persists data across user interactions
    if 'qa_bot' not in st.session_state:
        st.session_state.qa_bot = DocumentQABot()
    
    # Initialize chat message history in session state
    # This stores the conversation between user and bot
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Create sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Upload Document")
        
        # API Key input field
        # Users need to provide their own OpenAI API key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",                    # Hide the key as user types
            help="Enter your OpenAI API key"   # Tooltip text
        )
        
        # Set the API key as environment variable if provided
        # This is how the OpenAI library reads the key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            # Initialize OpenAI components now that we have the API key
            st.session_state.qa_bot.setup_openai_components(api_key)
        
        # File upload widget
        # Restricts to PDF files only for this demo
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",                                           # Only allow PDF files
            help="Upload a PDF document to ask questions about"  # Helper text
        )
        
        # Document processing section
        # Only show process button if both file and API key are provided
        if uploaded_file and api_key:
            # Check if OpenAI components are properly initialized
            if not st.session_state.qa_bot.embeddings or not st.session_state.qa_bot.llm:
                st.info("Initializing OpenAI components...")
                st.session_state.qa_bot.setup_openai_components(api_key)
                
            if st.button("Process Document", type="primary"):
                # Show spinner during processing (can take 30+ seconds for large docs)
                with st.spinner("Processing document..."):
                    # Step 1: Extract and chunk the PDF content
                    chunks = st.session_state.qa_bot.load_and_process_pdf(uploaded_file)
                    
                    if chunks:
                        # Step 2: Create embeddings and FAISS index
                        st.session_state.qa_bot.create_vectorstore(chunks)
                        
                        # Step 3: Set up the QA chain for answering questions
                        st.session_state.qa_bot.setup_qa_chain()
                        
                        # Clear any previous conversation when processing new document
                        st.session_state.messages = []
                        
                        st.success("Document processed successfully! You can now ask questions.")
        
        # Show warning if file uploaded without API key
        elif uploaded_file and not api_key:
            st.warning("Please enter your OpenAI API key to process the document.")
        
        # Display information about the uploaded file
        if uploaded_file:
            st.subheader("Document Info")
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")  # Convert bytes to KB
    
    # Main chat interface area
    st.header("💬 Ask Questions")
    
    # Display all previous messages in the conversation
    # This creates a ChatGPT-like interface where users can see chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # "user" or "assistant"
            st.markdown(message["content"])
            
            # Display source documents for assistant responses
            # This provides transparency about which parts of the document were used
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 Sources"):  # Collapsible section
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(source)
                        st.markdown("---")  # Visual separator
    
    # Chat input widget for new questions
    # This creates the input field at the bottom of the chat
    if prompt := st.chat_input("Ask a question about the document..."):
        # Validate prerequisites before processing question
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not st.session_state.qa_bot.qa_chain:
            st.error("Please upload and process a document first!")
        else:
            # Add user message to conversation history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display bot response
            with st.chat_message("assistant"):
                # Show thinking indicator while processing
                with st.spinner("Thinking..."):
                    # Call the DocumentQABot to get an answer
                    response = st.session_state.qa_bot.ask_question(prompt)
                    
                    # Handle error responses
                    if "error" in response:
                        st.error(response["error"])
                    else:
                        # Extract the answer from the response
                        answer = response.get("result", "Sorry, I couldn't find an answer.")
                        st.markdown(answer)
                        
                        # Extract source documents for transparency
                        # Show users which parts of the document were used to generate the answer
                        sources = []
                        if "source_documents" in response:
                            for doc in response["source_documents"]:
                                # Truncate long source text for readability
                                sources.append(doc.page_content[:200] + "...")
                        
                        # Display sources in expandable section
                        if sources:
                            with st.expander("📖 Sources"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.markdown(source)
                                    st.markdown("---")
                        
                        # Add assistant response to conversation history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
    
    # Example questions section
    # Provides quick-start buttons for common queries
    if st.session_state.qa_bot.qa_chain:
        st.subheader("💡 Example Questions")
        col1, col2, col3 = st.columns(3)  # Create three columns for buttons
        
        # Pre-defined example questions to help users get started
        with col1:
            if st.button("What is this document about?"):
                # Add question to messages and trigger rerun to process it
                st.session_state.messages.append({"role": "user", "content": "What is this document about?"})
                st.rerun()
        
        with col2:
            if st.button("Summarize the main points"):
                st.session_state.messages.append({"role": "user", "content": "Summarize the main points"})
                st.rerun()
        
        with col3:
            if st.button("What are the key findings?"):
                st.session_state.messages.append({"role": "user", "content": "What are the key findings?"})
                st.rerun()

# Application entry point
if __name__ == "__main__":
    # Handle missing dependencies gracefully
    # This provides clear error messages if packages aren't installed
    try:
        main()  # Run the Streamlit app
    except ImportError as e:
        st.error(f"""
        Missing required package: {e}
        
        Please install the required packages:
        ```bash
        pip install streamlit langchain openai faiss-cpu pypdf
        ```
        """)