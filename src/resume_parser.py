import pdfplumber
from docx import Document

def parse_resume(uploaded_file):
    """
    Extract text from PDF or DOCX resume
    """
    text = ""

    if uploaded_file is None:
        return text

    file_name = uploaded_file.name.lower()

    # -------- PDF --------
    if file_name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    # -------- DOCX --------
    elif file_name.endswith(".docx"):
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text.strip()

