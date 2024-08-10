from sys import path
from dotenv import dotenv_values
from collections import OrderedDict
from typing import List

from langchain_milvus import Milvus
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
                          model=_env_values["LLM_model_name"])
    _milvus: Milvus = Milvus(
        embedding_function=_embedding,
        connection_args={"uri": _env_values["milvus_uri"]},
        collection_name=_env_values["collection_name"],
        drop_old=True,
    )

    def __init__(self, prompt_template: str = _prompt_template, limit: int = 3):
        self._rag_prompt: PromptTemplate = PromptTemplate.from_template(prompt_template)
        self._retriever = self.__class__._milvus.as_retriever(search_type="similarity", search_kwargs={"k": limit})

        self._rag_chain = (
                {"context": self._retriever | self._format_doc, "question": RunnablePassthrough()}
                | self._rag_prompt
                | self.__class__._llm
                | StrOutputParser()
        )

    def get_response(self, query: str, similar_contexts: List[str]) -> str:
        """
        Get response from LLM model.

        This method using the chain that we create it constructor calls the model and return the answer.

        Parameters:
        query (str): user question without embeddings.
        similar_contexts (List[str]): documents that are similar to the query that user entered.

        Returns:
        str: output of chain invoke
        """

        return self._rag_chain.invoke({"context": similar_contexts, "question": query})

    def _search_docs(self, query: str) -> List[str]:
        """
        search similar documents and get the contents.

        This method using the Milvus retriever find most similar contents using the user embedding model
        from database and get their contents.

        Parameters:
        query (str): user question without embeddings.

        Returns:
        List[str]: list of contents that are most related to the user question.
        """

        similar_documents: List[Document] = self._retriever.invoke(query)
        contexts: List[str] = [document.page_content for document in similar_documents]
        return contexts

    @staticmethod
    def _format_doc(docs: List[Document]) -> str:
        """
        Joins page_content of each element using \n\n.

        This method get searched documents and convert them to a literal string using \n\n join.

        Parameters:
        docs (List[Documents]): List of searched documents.

        Returns:
        str: output of joins on the page contents.
        """

        return "\n\n".join(doc.page_content for doc in docs)
