from sentence_transformers import SentenceTransformer

class Vectorizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def vectorize(self, docs):
        return self.model.encode(docs)

    @staticmethod
    def check_model_name(model_name: str) -> bool:
        try:
            SentenceTransformer(model_name)
            return True
        except:
            return False
