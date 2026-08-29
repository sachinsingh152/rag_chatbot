import re

def recursive_character_text_split(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splits text recursively into chunks using a list of separators.
    This ensures we don't break up paragraphs or sentences if possible.
    """
    # The separators we will try in order
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    # If the text is already small enough, return it
    if len(text) <= chunk_size:
        return [text]
        
    return _split_text(text, separators, chunk_size, chunk_overlap)

def _split_text(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    # Find the appropriate separator to use
    separator = separators[-1] # default to ""
    for s in separators:
        if s == "":
            separator = s
            break
        if s in text:
            separator = s
            break
            
    # Split the text
    if separator:
        splits = text.split(separator)
    else:
        # If separator is empty string, split by character
        splits = list(text)
        
    # Re-merge the splits into chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for split in splits:
        split_len = len(split) + (len(separator) if current_length > 0 else 0)
        
        # If adding this split exceeds chunk size and we already have content
        if current_length + split_len > chunk_size and current_chunk:
            # Join the current chunk and add to chunks
            chunk_text = separator.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start a new chunk, trying to maintain overlap
            overlap_text = []
            overlap_length = 0
            
            # Work backwards from current_chunk to find overlap elements
            for item in reversed(current_chunk):
                item_len = len(item) + (len(separator) if overlap_length > 0 else 0)
                if overlap_length + item_len > chunk_overlap and overlap_text:
                    break
                overlap_text.insert(0, item)
                overlap_length += item_len
                
            current_chunk = overlap_text + [split]
            current_length = overlap_length + split_len
        else:
            current_chunk.append(split)
            current_length += split_len
            
    # Add the final chunk if it exists
    if current_chunk:
        chunk_text = separator.join(current_chunk)
        if chunk_text:
            chunks.append(chunk_text)
            
    # If a single chunk is still too big, we need to recursively split it with the next separator
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size and len(separators) > 1:
            # We need to go deeper (use remaining separators)
            next_separators = separators[separators.index(separator) + 1:]
            if next_separators:
                final_chunks.extend(_split_text(chunk, next_separators, chunk_size, chunk_overlap))
            else:
                final_chunks.append(chunk) # Can't split further
        else:
            final_chunks.append(chunk)
            
    return final_chunks


