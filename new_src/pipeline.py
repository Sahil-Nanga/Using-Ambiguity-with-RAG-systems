import os 
from .embedder import Embedder
from .cleaner import Cleaner
from .ambiguous import Ambiguous
from .retriever import Retriever
from .indexer import Indexer
from .generator import Generator
import numpy as np

class RagPipeline():
    def __init__(self):
        self.embedder = Embedder('sentence-transformers/all-MiniLM-L6-v2')
        self.cleaner = Cleaner()
        self.ambiguous  = Ambiguous()
        self.indexer = None
        if not os.path.exists("embeddings\\faiss_index.bin"):
            self.indexer = Indexer().add_documents()
            print("Indexer Finished")
        self.retriever = Retriever()
        # self.generator = Generator("deepseek-r1:1.5b")

    def ask_query(self, query, make_query_ambiguous):
        clean_query = self.cleaner.clean_text(query)  # Cleans user query
        query_embedding = self.embedder.encode([clean_query])
        retrieved_indices = self.retriever.search(query_embedding, top_k=20)  # Retrieve similar documents
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if make_query_ambiguous:
            variations = self.ambiguous.make_ambiguous(clean_query)
            versions = {}
            
            for variation in list(variations.keys()):
                
                variation_embedding = self.embedder.encode(variations[variation])
                if len(variation_embedding.shape) == 1:
                    variation_embedding = variation_embedding.reshape(1, -1)
                variation_indices = self.retriever.search(variation_embedding, top_k=20)
                
                versions[variation] = variation_indices#np.concatenate((retrieved_indices[:10], variation_indices))

            
            
            versions["orginal"] = retrieved_indices
            return versions
        return retrieved_indices
    
    def retrieve_documents(self, indices):
        docs, ids = self.retriever.get_documents(indices)
        return ids
