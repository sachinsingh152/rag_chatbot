import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import hashlib
import os

DATA_DIR = os.getenv('DATA_DIR', '.')
PERSIST_DIR = os.path.join(DATA_DIR, 'chroma_db')

class VectorStore:
    def __init__(self, persist_directory: str = PERSIST_DIR, collection_name: str = "rag_collection"):
        """
        Initializes the VectorStore with a SentenceTransformer model and a ChromaDB client.
        """
        print("Loading embedding model (this might take a moment the first time)...")
        # all-MiniLM-L6-v2 is fast, lightweight, and suitable for local CPU execution
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Loading cross-encoder re-ranking model...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
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

    def add_documents(self, filename: str, text: str, chunks: list[str], username: str) -> tuple[bool, str]:
        """
        Embeds and stores chunks if the file hasn't been added already.
        Returns a tuple of (success_boolean, message).
        """
        if not chunks:
            return False, f"No chunks provided for {filename}."

        # Check if file already exists in metadata to prevent duplicates
        existing_docs = self.collection.get(
            where={"$and": [{"filename": filename}, {"username": username}]},
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
            chunk_id = f"{username}_{filename}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "chunk_index": i,
                "file_hash": file_hash,
                "username": username
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
        
    def retrieve(self, query: str, top_k: int = 4, selected_files: list[str] = None, username: str = None) -> list[dict]:
        """
        Retrieves the top_k most relevant chunks for a given query.
        Returns a list of dictionaries containing the chunk, metadata, and distance.
        """
        if not query.strip():
            return []

        # Embed the query using the same model
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        where_clause = None
        if username and selected_files:
            if len(selected_files) == 1:
                where_clause = {"$and": [{"username": username}, {"filename": selected_files[0]}]}
            else:
                where_clause = {"$and": [{"username": username}, {"filename": {"$in": selected_files}}]}
        elif username:
            where_clause = {"username": username}
        elif selected_files:
            if len(selected_files) == 1:
                where_clause = {"filename": selected_files[0]}
            else:
                where_clause = {"filename": {"$in": selected_files}}
        
        # Fetch more candidates for re-ranking (e.g., 4x the top_k, maxing at a reasonable number)
        fetch_k = min(top_k * 4, 20)
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        initial_chunks = []
        # Check if we got results
        if results and results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                initial_chunks.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
                
        if not initial_chunks:
            return []
            
        # Cross-Encoder Re-ranking
        # Pair each document with the query
        cross_inp = [[query, chunk["document"]] for chunk in initial_chunks]
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        # Add scores to chunks and sort
        for i in range(len(initial_chunks)):
            initial_chunks[i]["cross_score"] = float(cross_scores[i])
            
        # Sort descending by cross_score (higher is better)
        initial_chunks.sort(key=lambda x: x["cross_score"], reverse=True)
        
        # Return only the top_k
        return initial_chunks[:top_k]
        
    def delete_file(self, filename: str, username: str) -> str:
        """Deletes all chunks associated with a specific file and user from the vector store."""
        try:
            self.collection.delete(where={"$and": [{"filename": filename}, {"username": username}]})
            return f"Successfully deleted '{filename}' from your vector store."
        except Exception as e:
            return f"Error deleting file: {str(e)}"
            
    def clear_vector_store(self, username: str = None) -> str:
        """Clears all documents from the collection for a specific user, or all if no user provided."""
        if username:
            self.collection.delete(where={"username": username})
            return f"Vector store cleared successfully for user {username}."
        else:
            collection_name = self.collection.name
            self.client.delete_collection(collection_name)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return "Vector store cleared successfully for all users."
        
    def get_indexed_files(self, username: str = None) -> list[str]:
        """Returns a list of unique filenames currently in the index for a specific user."""
        # We only need the metadata to extract unique filenames
        where_clause = {"username": username} if username else None
        results = self.collection.get(where=where_clause, include=["metadatas"])
        if not results or not results.get('metadatas'):
            return []
            
        filenames = set()
        for meta in results['metadatas']:
            if 'filename' in meta:
                filenames.add(meta['filename'])
            
        return sorted(list(filenames))


