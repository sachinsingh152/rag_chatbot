import io
import fitz  # PyMuPDF
import docx

def extract_text_from_pdf(file_input) -> str:
    """Extracts text from a PDF file. `file_input` can be a path or a file-like object."""
    text = ""
    try:
        # Check if it's a string (path) or a bytes-like object
        if isinstance(file_input, str):
            doc = fitz.open(file_input)
        else:
            # It's a file-like object (e.g., from Streamlit)
            doc = fitz.open(stream=file_input.read(), filetype="pdf")
            
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
            
        if not text.strip():
            # Basic check for scanned PDF
            return "WARNING_NO_TEXT: This PDF appears to be an image or scanned document. Text extraction yielded no content."
    except Exception as e:
        return f"ERROR: Failed extracting PDF: {str(e)}"
        
    return text.strip()

def extract_text_from_docx(file_input) -> str:
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(file_input)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"ERROR: Failed extracting DOCX: {str(e)}"
    return text.strip()

def extract_text_from_txt(file_input) -> str:
    """Extracts text from a TXT file."""
    try:
        if isinstance(file_input, str):
            with open(file_input, "r", encoding="utf-8") as f:
                return f.read()
        else:
            # Streamlit UploadedFile bytes
            return file_input.read().decode("utf-8")
    except Exception as e:
        return f"ERROR: Failed extracting TXT: {str(e)}"

def extract_text(file_input, file_type: str) -> str:
    """Router function to extract text based on file type."""
    file_type = file_type.lower()
    if file_type == "pdf":
        return extract_text_from_pdf(file_input)
    elif file_type == "docx":
        return extract_text_from_docx(file_input)
    elif file_type == "txt":
        return extract_text_from_txt(file_input)
    else:
        return f"ERROR: Unsupported file type: {file_type}"


