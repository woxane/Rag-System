from typing import Tuple, Type

from pymilvus import MilvusClient



class MilvusHandler:
    def __init__(self, collection_name, dimensions, milvus_uri):
        self.milvus_client = MilvusClient(milvus_uri)
        self.collection_name = collection_name
        self.dimensions = dimensions

        if self.milvus_client.has_collection(collection_name=self.collection_name):
            self.milvus_client.drop_collection(collection_name=self.collection_name)
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimensions,
        )

    def save_vectors(self, vectors, chunks, file_id):
        data = [
            {"id": i, "vector": vectors[i], "text": chunks[i], "subject": "history", "file_group": file_id}
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


    def delete_vectors(self, file_id):
        self.milvus_client.delete(
            collection_name=self.collection_name,
            filter=f"file_group == {file_id}",
            )


    @staticmethod
    def check_milvus_uri(milvus_uri: str) -> str:
        try:
            MilvusClient(uri=milvus_uri)
            return ""
        except Exception as e:
            return str(e)
