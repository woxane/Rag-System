from sys import path
from dotenv import dotenv_values
from collections import OrderedDict

from langchain_milvus import Milvus
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

path.append('../')

from utils.document_processor import DocumentProcessor

dotenv_path = '.env'


class Chatbot:
    _env_values: OrderedDict = dotenv_values(dotenv_path)

    _documentProcessor: DocumentProcessor = DocumentProcessor(chunk_size=int(_env_values["chunk_size"]),
                                                              chunk_overlap=int(_env_values["chunk_overlap"]))
    _embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(_env_values['embedding_model_name'])
    _llm: OpenAI = OpenAI(base_url=_env_values["openAI_base_url"],
                          api_key=_env_values["openAI_api_key"],
                          model_name=_env_values["LLM_model_name"])
    _milvus: Milvus = Milvus(
        embedding_function=_embedding,
        connection_args={"uri": _env_values["milvus_uri"]},
    )
