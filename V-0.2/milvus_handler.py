from typing import Tuple, Type

from pymilvus import MilvusClient, DataType
from uuid import uuid4


class MilvusHandler:
    def __init__(self, collection_name, dimensions, milvus_uri, chunk_size=256):
        self.milvus_client = MilvusClient(milvus_uri)
        self.collection_name = collection_name
        self.dimensions = dimensions


        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimensions)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2*chunk_size)
        schema.add_field(field_name="file_id", datatype=DataType.VARCHAR, max_length=64)

        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index("id")
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={ "nlist": 128}
        )
        index_params.add_index("text")
        index_params.add_index("file_id")


        if self.milvus_client.has_collection(collection_name=self.collection_name):
            self.milvus_client.drop_collection(collection_name=self.collection_name)
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def save_vectors(self, vectors, chunks, file_id):
        data = [
            {"id": str(uuid4()), "vector": vectors[i], "text": chunks[i], "file_id": file_id}
            for i in range(len(vectors))
        ]

        print(data)
        self.milvus_client.insert(collection_name=self.collection_name, data=data)

    def search_vectors(self, query_vector, top_k=3):
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=query_vector,
            limit=top_k,
            output_fields=["text", "file_id"],
        )
        return results


    def delete_vectors(self, file_id):
        self.milvus_client.delete(
            collection_name=self.collection_name,
            filter=f"file_id == '{file_id}'",
            )


    @staticmethod
    def check_milvus_uri(milvus_uri: str) -> str:
        try:
            MilvusClient(uri=milvus_uri)
            return ""
        except Exception as e:
            return str(e)
