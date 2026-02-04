# lightweight_rag_engine.py
"""
Lightweight RAG Engine using TfidfVectorizer and sklearn's cosine_similarity.
This eliminates the need for sentence-transformers and HuggingFaceEmbeddings.
Memory footprint: ~50-100MB vs 6GB with sentence-transformers.
"""

import logging
import re
import pathlib
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import docx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightweightTextEncoder:
    """
    Lightweight text encoder using TfidfVectorizer.
    No pre-trained models, no heavy dependencies.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Initialize the encoder.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: (1, 2) for unigrams and bigrams
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        self.is_fitted = False
        self.chunks = []
        self.chunk_matrix = None
        
    def fit_on_chunks(self, chunks: List[str]) -> None:
        """
        Fit the vectorizer on the provided chunks.
        
        Args:
            chunks: List of text chunks to fit on
        """
        if not chunks:
            logger.warning("No chunks provided for fitting.")
            return
            
        self.chunks = chunks
        self.chunk_matrix = self.vectorizer.fit_transform(chunks)
        self.is_fitted = True
        logger.info(f"Encoder fitted on {len(chunks)} chunks. Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            numpy array of shape (len(texts), n_features)
        """
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call fit_on_chunks() first.")
        
        return self.vectorizer.transform(texts).toarray()
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.
        
        Args:
            query: Query text
            
        Returns:
            numpy array of shape (1, n_features)
        """
        return self.encode([query])


class LightweightRAGEngine:
    """
    Lightweight RAG Engine using TfidfVectorizer for embeddings.
    Provides chunking, encoding, and retrieval functionality.
    """
    
    def __init__(self, chunk_size: int = 1200, overlap: int = 200, max_features: int = 5000):
        """
        Initialize the RAG engine.
        
        Args:
            chunk_size: Size of text chunks in characters
            overlap: Overlap between chunks in characters
            max_features: Maximum vocabulary size for TfidfVectorizer
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = LightweightTextEncoder(max_features=max_features)
        self.chunks = []
        self.chunk_matrix = None
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        n = len(text)
        
        while start < n:
            end = min(start + self.chunk_size, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            start = max(end - self.overlap, 0)
        
        logger.info(f"Text chunked into {len(chunks)} chunks (chunk_size={self.chunk_size}, overlap={self.overlap})")
        return chunks
    
    def build_index(self, text: str) -> None:
        """
        Build the RAG index from raw text.
        
        Args:
            text: Raw text to index
        """
        self.chunks = self.chunk_text(text)
        self.encoder.fit_on_chunks(self.chunks)
        self.chunk_matrix = self.encoder.chunk_matrix
        logger.info(f"RAG index built with {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 6) -> List[str]:
        """
        Retrieve top-K most similar chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of top-K relevant chunks
        """
        if not self.encoder.is_fitted:
            raise ValueError("RAG engine not initialized. Call build_index() first.")
        
        query_vector = self.encoder.encode_query(query)
        
        # Compute cosine similarity between query and all chunks
        similarities = cosine_similarity(query_vector, self.chunk_matrix)[0]
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter out very low similarity scores (optional, improves quality)
        result_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Threshold to avoid noise
                result_chunks.append(self.chunks[idx])
        
        logger.info(f"Retrieved {len(result_chunks)} chunks for query (top_k={top_k})")
        return result_chunks
    
    def generate_rag_prompt(self, query: str, top_k: int = 6) -> str:
        """
        Generate a RAG prompt with retrieved context.
        
        Args:
            query: User query
            top_k: Number of context chunks
            
        Returns:
            Formatted prompt for LLM
        """
        retrieved_chunks = self.retrieve(query, top_k=top_k)
        
        rag_context = "\n\n--- RAG CONTEXT ---\n" + \
                      "\n\n".join(f"[CHUNK {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)) + \
                      "\n--- END RAG CONTEXT ---\n"
        
        prompt = f"""You are a helpful assistant. Please provide a detailed response based on the context below. 
                Respond only with the single best example abap object details and a sample code snippet to use it. 
                Please provide the sample code snippet in ABAP 7.5 version.
                And please respond back in the chat itself.

                Rag_context: {rag_context}

                Question: {query}

                """
        
        return prompt


# ==================== File Loading Utilities ====================

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_text(fs_path: str) -> str:
    """
    Load text from file system (supports .txt, .md, .docx).
    
    Args:
        fs_path: Path to file
        
    Returns:
        Text content
    """
    p = pathlib.Path(fs_path).expanduser().resolve()
    
    if not p.exists():
        raise FileNotFoundError(f"File not found: {fs_path}")
    
    if p.suffix.lower() in [".txt", ".md", ".json"]:
        return p.read_text(encoding="utf-8", errors="ignore")
    
    # if p.suffix.lower() == ".docx":
    #     document = docx.Document(str(p))
    #     return "\n".join([para.text for para in document.paragraphs])
    
    # Fallback: try reading as text
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Unsupported file format: {p.suffix}. Use .txt, .md, or .docx. Error: {e}")


# ==================== Example Usage ====================

# if __name__ == "__main__":
#     # Example 1: RAG on file system text
#     print("\n=== Example 1: RAG on File System Text ===")
#     try:
#         # Load a document
#         fs_text = load_fs_text("path/to/your/document.txt")  # Change this path
#         fs_text = normalize_whitespace(fs_text)
        
#         # Initialize RAG engine
#         rag_engine = LightweightRAGEngine(chunk_size=1200, overlap=200, max_features=5000)
#         rag_engine.build_index(fs_text)
        
#         # Example query
#         query = "What is the main topic of this document?"
#         prompt = rag_engine.generate_rag_prompt(query, top_k=6)
#         print(prompt)
        
#     except FileNotFoundError as e:
#         print(f"File error: {e}")
    
#     # Example 2: Direct text processing
#     print("\n=== Example 2: Direct Text Processing ===")
#     sample_text = """
#     Machine learning is a subset of artificial intelligence.
#     Deep learning uses neural networks with multiple layers.
#     Natural language processing helps computers understand text.
#     Computer vision enables machines to interpret images.
#     """
    
#     rag_engine = LightweightRAGEngine(chunk_size=200, overlap=30)
#     rag_engine.build_index(sample_text)
    
#     query = "What is machine learning?"
#     result = rag_engine.retrieve(query, top_k=3)
#     print(f"\nQuery: {query}")
#     print(f"Retrieved chunks: {result}")