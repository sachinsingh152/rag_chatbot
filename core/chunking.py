from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_character_text_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Splits text recursively into chunks using LangChain's RecursiveCharacterTextSplitter.
    This ensures we don't break up paragraphs or sentences if possible.
    
    Default chunk_size increased to 1000 and overlap to 200 to preserve semantic meaning 
    and prevent definitions from being severed from their examples.
    """
    # The standard separators for RecursiveCharacterTextSplitter
    # It tries to split on paragraphs first, then newlines, then sentences, then words.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks
