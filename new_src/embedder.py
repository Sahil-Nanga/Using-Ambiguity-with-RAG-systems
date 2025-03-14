
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts):#returns encoded data in a numpy array with float32 datatype
        return self.model.encode(texts, convert_to_numpy=True)