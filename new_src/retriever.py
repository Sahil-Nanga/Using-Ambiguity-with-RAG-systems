import os
import faiss
import numpy as np
from datasets import load_dataset
from .embedder import Embedder  # Ensure correct import

class Retriever:
    def __init__(self, index_file=os.path.join("embeddings", "faiss_index.bin")):
        self.embedder = Embedder('sentence-transformers/all-MiniLM-L6-v2')
        self.index_file = index_file
        self.index = None
        self.load_index()

    def load_index(self):
        self.index = faiss.read_index(self.index_file)
        if not isinstance(self.index, faiss.IndexIDMap):
            self.index = faiss.IndexIDMap(self.index)
        

    def search(self, query_embedding, top_k=5):
        """Retrieve top-k most relevant document IDs and distances"""

        distances, ids = self.index.search(query_embedding, top_k)  # Perform FAISS search

        return ids[0]
    def get_documents(self,ids):
        dataset = load_dataset("enelpol/rag-mini-bioasq", 'text-corpus')['test']
        filtered_documents = [item['passage'] for item in dataset if item['id'] in ids]
        return filtered_documents,ids