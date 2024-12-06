import streamlit as st
import os
from PIL import Image
import PyPDF2
import pytesseract
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if groq_api_key is None:
    st.error("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")
    st.stop()
if google_api_key is None:
    st.error("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("Document Retrieval System 📄🔍")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to process documents
def process_documents(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                docs.append(page.extract_text())
        elif uploaded_file.type in ["image/png", "image/jpeg"]:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            docs.append(text)
    return docs

# Function to generate keyword suggestions using TF-IDF
def generate_suggestions(docs, top_n=5):
    suggestions = []
    for doc in docs:
        tfidf = TfidfVectorizer(stop_words='english', max_features=top_n)
        tfidf_matrix = tfidf.fit_transform([doc])
        keywords = tfidf.get_feature_names_out()
        suggestions.append(list(keywords))
    return suggestions


# File uploader
uploaded_files = st.file_uploader("Upload up to 3 Documents (PDF, JPG, PNG)", type=["pdf", "png", "jpg"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 3:
        st.warning("You can upload up to 3 documents only. Processing the first 3 documents.")
        uploaded_files = uploaded_files[:3]

    docs = process_documents(uploaded_files)
    st.success("Documents processed successfully!")

    suggestions = generate_suggestions(docs)
    st.write("Here are 5 unique keyword suggestions per document:")
    for i, doc_suggestions in enumerate(suggestions, start=1):
        st.write(f"Document {i} Suggestions: {doc_suggestions}")

# Search functionality
with st.form(key='search_form'):
    selected_doc_index = st.selectbox("Select Document to Search In", list(range(1, len(uploaded_files) + 1)))
    search_query = st.text_input("Enter your search keyword or question:")
    submit_button = st.form_submit_button(label='Search')

if submit_button and search_query:
    selected_doc = docs[selected_doc_index - 1]
    retriever = FAISS.from_texts([selected_doc], GoogleGenerativeAIEmbeddings(model="models/embedding-001")).as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': search_query})
    st.write("Search Results:")
    st.write(response['answer'])

    with st.expander("Matching Document Context"):
        for i, context in enumerate(response["context"]):
            st.write(f"Context {i + 1}: {context}")
            st.write("--------------------------------")

# Download functionality
if uploaded_files:
    st.write("Download Individual Documents:")
    for i, file in enumerate(uploaded_files, start=1):
        st.download_button(label=f"Download Document {i}", data=file.getvalue(), file_name=file.name, mime=file.type)

# Sidebar
with st.sidebar:
    st.title("📚 Document Retrieval System")
    st.markdown("""
        ## Features:
        - Upload up to 3 documents (PDF, PNG, JPG)
        - Automatically generates 5 keyword suggestions per document
        - Search functionality with accurate keyword matching
        - Download individual documents
        - Responsive and user-friendly design
    """)
    st.write("Made with ❤️ by Harshit")

