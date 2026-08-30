import io
import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image

def extract_text_from_image(file_input) -> str:
    """Extracts text from an image file using Tesseract OCR."""
    try:
        if isinstance(file_input, str):
            img = Image.open(file_input)
        else:
            # It's a file-like object (e.g., from Streamlit)
            img = Image.open(io.BytesIO(file_input.read()))
            
        # Ensure image is in a format Tesseract handles well (RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"ERROR: Failed extracting text from image: {str(e)}"

def extract_text_from_pdf(file_input) -> str:
    """
    Extracts text from a PDF file. 
    Performs hybrid extraction: reads native text, then uses OCR on embedded images.
    """
    text = ""
    try:
        # Check if it's a string (path) or a bytes-like object
        if isinstance(file_input, str):
            doc = fitz.open(file_input)
        else:
            # Reset pointer just in case
            file_input.seek(0)
            doc = fitz.open(stream=file_input.read(), filetype="pdf")
            
        for page_num, page in enumerate(doc):
            # 1. Extract standard selectable text
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
                
            # 2. Extract embedded images and run OCR
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                try:
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    
                    ocr_text = pytesseract.image_to_string(pil_img).strip()
                    if ocr_text:
                        text += f"\n[Embedded Image {img_index+1} OCR on Page {page_num+1}]:\n{ocr_text}\n"
                except Exception as img_e:
                    # Silently skip images that fail to process to prevent crashing the whole doc
                    pass
            
        if not text.strip():
            # Fallback if no text or images were successfully parsed
            return "WARNING_NO_TEXT: This PDF appears to be empty or unreadable."
            
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
            file_input.seek(0)
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
    elif file_type in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(file_input)
    else:
        return f"ERROR: Unsupported file type: {file_type}"
