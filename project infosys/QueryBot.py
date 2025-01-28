import os
from dotenv import load_dotenv
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
import tempfile
import warnings

# Suppress all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress LangChain-specific warnings
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and configurations from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists, and clear it or connect
try:
    if PINECONE_INDEX_NAME in pc.list_indexes():
        print(f"Index {PINECONE_INDEX_NAME} already exists.")
    else:
        print(f"Creating new index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # Dimension for the embedding model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )
except Exception as e:
    print(f"Error with Pinecone Index: {e}")

# Connect to the existing index
index = pc.Index(PINECONE_INDEX_NAME)

# Function to load data from PDFs in a directory
def load_data(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    data = loader.load()
    return data

# Function to split text into chunks
def text_split(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = splitter.split_documents(data)
    return text_chunks

# Download HuggingFace embeddings
def download_huggingface_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Initialize embeddings
embeddings = download_huggingface_embedding()

# Streamlit UI improvements
st.set_page_config(page_title="Medical AI Q&A Bot", page_icon="💉", layout="wide")
st.title("Welcome to the Medical AI Q&A Bot 🏥")
st.markdown(
    """
    Upload your medical PDF documents and ask any questions related to the content.
    Our AI will assist you in finding answers directly from your documents. 
    Let's make your healthcare information more accessible!
    """
)

# Layout with columns (Left: Upload Area, Right: Query Area)
col1, col2 = st.columns([1, 2])  # Column ratio (1: Upload area, 2: Query area)

# File upload section on the left column
with col1:
    st.markdown(
        """
        <style>
        .upload-section {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .upload-section h3 {
            color: #3b3b3b;
        }
        .upload-section p {
            color: #777;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="upload-section"><h3>Upload Medical PDF Documents</h3><p>Please upload your PDF files containing medical information below:</p></div>', unsafe_allow_html=True)

    # File upload section
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.success("Documents uploaded successfully! 📚")

        # Show a progress bar while processing
        with st.spinner("Processing your documents..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())

                # Load documents
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                # Split text into chunks
                text_chunks = text_split(docs)

            #st.info(f"Processed {len(text_chunks)} chunks of text from the uploaded documents. 🔄")

            # Use Pinecone to add embeddings
            vectorstore = LangChainPinecone.from_documents(
                text_chunks,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME
            )

            st.success("Text successfully embedded into Pinecone! ✨")

# Prompt area (query input) on the right column
with col2:
    # Set up the prompt template for medical questions
    prompt_template = """
    Use the given context to provide an appropriate answer in atleast 3 lines  to the user's medical question.
    If you don't know the answer, say you don't know, but don't fabricate an answer.
    Context: {context}
    Question: {question}
    Only return the appropriate answer.
    """

    # Initialize Gemini API
    llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GEMINI_API_KEY, temperature=0.7, top_p=0.85,max_tokens=500)

    # Initialize prompt and chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Initialize the RetrievalQA chain only after vectorstore is defined
    if 'vectorstore' in locals():  # Ensure the vectorstore is initialized first
        qa_chain = RetrievalQA.from_chain_type(
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            llm=llm,
            chain_type_kwargs={"prompt": prompt},
        )

        # User query input for medical questions
        query = st.text_input("Ask a medical question about your documents:", "")

        if query:
            with st.spinner("Fetching answer..."):
                answer = qa_chain.run(query)
                st.subheader("Answer 💡")
                st.write(answer)

# Footer
st.markdown(
    """
    ---
    📢 **Medical AI Q&A Bot** • Powered by LangChain, Pinecone, and Google Generative AI.
    """
)
