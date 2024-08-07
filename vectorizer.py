from sentence_transformers import SentenceTransformer

class Vectorizer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def vectorize(self, docs):
        return self.model.encode(docs)

    @staticmethod
    def check_model_name(model_name: str) -> str:
        try:
            SentenceTransformer(model_name)
            return ""
        except Exception as e:
            return str(e)
