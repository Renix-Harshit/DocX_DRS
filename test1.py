import streamlit as st
import os
import re
import pytesseract
import PyPDF2
import pandas as pd
from docx import Document
from PIL import Image
from collections import Counter

# Set up Streamlit app
st.title("Document Keyword Suggestion & Search Tool 📄🔍")

# Helper functions
def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() for page in pdf_reader.pages])

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    doc = Document(file)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_excel(file):
    """Extract text from an Excel file."""
    excel_data = pd.ExcelFile(file)
    text = ""
    for sheet in excel_data.sheet_names:
        df = excel_data.parse(sheet)
        text += df.to_string(index=False, header=True)
    return text

def extract_text_from_image(file):
    """Extract text from an image using OCR."""
    image = Image.open(file)
    return pytesseract.image_to_string(image)

def generate_keyword_suggestions(text, num_suggestions=10):
    """Generate keyword suggestions from text."""
    tokens = re.findall(r'\b[A-Za-z0-9_]+\b', text)
    keyword_counts = Counter(tokens)
    suggestions = [keyword[0] for keyword in keyword_counts.most_common(num_suggestions)]
    return suggestions

def search_in_text(text, query):
    """Search for a query in the given text."""
    return query.lower() in text.lower()

# File upload
uploaded_files = st.file_uploader(
    "Upload up to 3 documents (PDF, DOCX, Excel, or Image)", 
    type=["pdf", "docx", "xlsx", "jpg", "png"], 
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    if len(uploaded_files) > 3:
        st.warning("You can only upload up to 3 files. Processing the first 3 files.")
        uploaded_files = uploaded_files[:3]

    # Extract text from uploaded files
    documents = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            text = extract_text_from_excel(file)
        elif file.type in ["image/jpeg", "image/png"]:
            text = extract_text_from_image(file)
        else:
            text = ""
        documents.append({"name": file.name, "text": text, "file": file})

    # Generate and display keyword suggestions
    st.subheader("Keyword Suggestions")
    for idx, doc in enumerate(documents):
        st.write(f"**Suggestions for {doc['name']}:**")
        suggestions = generate_keyword_suggestions(doc["text"])
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
        doc["keywords"] = suggestions

    # Search functionality
    st.subheader("Search Documents")
    with st.form(key="search_form"):
        search_query = st.text_input("Enter a search query:")
        specific_file = st.selectbox("Select a specific file to search (optional):", ["All"] + [doc["name"] for doc in documents])
        search_button = st.form_submit_button(label="Search")

    if search_button and search_query:
        # Perform search
        results = []
        for doc in documents:
            if specific_file == "All" or doc["name"] == specific_file:
                if search_in_text(doc["text"], search_query):
                    results.append(doc)

        # Display results
        if results:
            st.success(f"Found matches in {len(results)} document(s):")
            for result in results:
                st.write(f"- **{result['name']}**")
                st.download_button(
                    label=f"Download {result['name']}",
                    data=result["file"].getvalue(),
                    file_name=result["name"]
                )
        else:
            st.error("No matches found.")

# Sidebar
with st.sidebar:
    st.title("ℹ️ About")
    st.write("""
    This tool allows you to:
    - Upload up to 3 documents (PDF, DOCX, Excel, Image)
    - Generate keyword suggestions for each document
    - Search for specific content across all or specific documents
    - Download documents containing search results
    """)
    st.write("Built with ❤️ by Harshit.")