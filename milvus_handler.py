from pymilvus import MilvusClient



class MilvusHandler:
    def __init__(self, collection_name, dimensions, milvus_uri="http://localhost:19530"):
        self.milvus_client = MilvusClient(milvus_uri)
        self.collection_name = collection_name
        self.dimensions = dimensions

    def save_vectors(self, vectors, chunks):
        if self.milvus_client.has_collection(collection_name=self.collection_name):
            self.milvus_client.drop_collection(collection_name=self.collection_name)
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimensions,
        )

        data = [
            {"id": i, "vector": vectors[i], "text": chunks[i], "subject": "history"}
            for i in range(len(vectors))
        ]

        print(data)
        self.milvus_client.insert(collection_name=self.collection_name, data=data)

    def search_vectors(self, query_vector, top_k=3):
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_vector,
            limit=top_k,
            output_fields=["text", "subject"],
        )
        return results
