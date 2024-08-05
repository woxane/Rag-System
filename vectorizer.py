from sentence_transformers import SentenceTransformer

class Vectorizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def vectorize(self, docs):
        return self.model.encode(docs)
