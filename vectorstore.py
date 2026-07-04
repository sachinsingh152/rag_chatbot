import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import os

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "rag_collection"):
        """
        Initializes the VectorStore with a SentenceTransformer model and a ChromaDB client.
        """
        print("Loading embedding model (this might take a moment the first time)...")
        # all-MiniLM-L6-v2 is fast, lightweight, and suitable for local CPU execution
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create the collection (using cosine similarity)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} 
        )
        
    def _get_file_hash(self, text: str) -> str:
        """Generate a hash for the document to avoid re-embedding."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def add_documents(self, filename: str, text: str, chunks: list[str]) -> tuple[bool, str]:
        """
        Embeds and stores chunks if the file hasn't been added already.
        Returns a tuple of (success_boolean, message).
        """
        if not chunks:
            return False, f"No chunks provided for {filename}."

        # Check if file already exists in metadata to prevent duplicates
        existing_docs = self.collection.get(
            where={"filename": filename},
            limit=1
        )
        
        if existing_docs and existing_docs['ids']:
            return False, f"File '{filename}' is already indexed."
            
        file_hash = self._get_file_hash(text)
        
        # Prepare data for Chroma
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        # Generate embeddings in batch for efficiency
        # This returns a numpy array which we convert to a list of lists for Chroma
        chunk_embeddings = self.embedding_model.encode(chunks).tolist()
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "chunk_index": i,
                "file_hash": file_hash
            })
            embeddings.append(chunk_embeddings[i])
            
        # Add to Chroma DB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        return True, f"Successfully indexed {len(chunks)} chunks from '{filename}'."
        
    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Retrieves the top_k most relevant chunks for a given query.
        Returns a list of dictionaries containing the chunk, metadata, and distance.
        """
        if not query.strip():
            return []

        # Embed the query using the same model
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_chunks = []
        # Check if we got results
        if results and results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                retrieved_chunks.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]  # Cosine distance (lower is more similar)
                })
                
        return retrieved_chunks
        
    def clear_vector_store(self) -> str:
        """Clears all documents from the collection."""
        collection_name = self.collection.name
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return "Vector store cleared successfully."
        
    def get_indexed_files(self) -> list[str]:
        """Returns a list of unique filenames currently in the index."""
        # We only need the metadata to extract unique filenames
        results = self.collection.get(include=["metadatas"])
        if not results or not results.get('metadatas'):
            return []
            
        filenames = set()
        for meta in results['metadatas']:
            if 'filename' in meta:
                filenames.add(meta['filename'])
            
        return sorted(list(filenames))


