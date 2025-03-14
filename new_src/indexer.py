import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from .embedder import Embedder

class Indexer:
    def __init__(self):
        self.embedder = Embedder('sentence-transformers/all-MiniLM-L6-v2')
        self.dataset = None  # Lazy loading
        self.index_file = os.path.join("embeddings", "faiss_index.bin")
        self.index = None

    def load_dataset(self):
        """Load dataset if not already loaded"""
        if self.dataset is None:
            self.dataset = load_dataset("enelpol/rag-mini-bioasq", 'text-corpus')['test']

    def initialize_index(self, dimension=384):
        """Initialize FAISS index with L2 (Euclidean) distance and ID mapping"""
        base_index = faiss.IndexFlatL2(dimension)  # Base index for vector search
        print("Base index created")

        self.index = faiss.IndexIDMap(base_index)  # Map vectors to IDs
        print("ID index created")
    def add_documents(self, batch_size=1000):
        """Embeds the dataset and adds vectors to FAISS index with IDs"""
        self.load_dataset()
        print("Dataset loaded")
        self.initialize_index()
        print("The index is created")
        texts = [text["passage"].replace("\n", "") for text in self.dataset]
        indices = np.array([int(text["id"]) for text in self.dataset], dtype=np.int64)  # FAISS requires int64 IDs

        for i in range(0, len(texts), batch_size):
            print(i)
            batch_texts = texts[i:i + batch_size]
            batch_indices = indices[i:i + batch_size]
            
            embeddings = self.embedder.encode(batch_texts)
            
            self.index.add_with_ids(embeddings, batch_indices)  # Add embeddings with IDs

        # Save FAISS index
        faiss.write_index(self.index, self.index_file)

        print(f"Added {len(texts)} documents to the vector database.")
