from sys import path
from dotenv import dotenv_values
from collections import OrderedDict
from typing import List

from langchain_milvus import Milvus
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

path.append('../')

from utils.document_processor import DocumentProcessor

dotenv_path = '.env'


class Chatbot:
    _prompt_template: str = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

    _env_values: OrderedDict = dotenv_values(dotenv_path)

    _documentProcessor: DocumentProcessor = DocumentProcessor(chunk_size=int(_env_values["chunk_size"]),
                                                              chunk_overlap=int(_env_values["chunk_overlap"]))
    _embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=_env_values['embedding_model_name'])
    _llm: OpenAI = OpenAI(base_url=_env_values["openAI_base_url"],
                          api_key=_env_values["openAI_api_key"],
                          model_name=_env_values["LLM_model_name"])
    _milvus: Milvus = Milvus(
        embedding_function=_embedding,
        connection_args={"uri": _env_values["milvus_uri"]},
    )

    def __init__(self, prompt_template=_prompt_template):
        self._rag_template: PromptTemplate = PromptTemplate.from_template(prompt_template)

    def _format_doc(self, docs: List[Document]) -> str:
        """
        Joins page_content of each element using \n\n.

        This method get searched documents and convert them to a literal string using \n\n join.

        Parameters:
        docs (List[Documents]): List of searched documents.

        Returns:
        str: output of joins on the page contents.
        """
        
        return "\n\n".join(doc.page_content for doc in docs)
