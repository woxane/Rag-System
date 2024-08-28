from sys import path
from dotenv import dotenv_values
from collections import OrderedDict
from typing import List, Iterator, Dict
from uuid import uuid4, UUID

from langchain_milvus import Milvus
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

path.append('../')

from utils.document_processor import DocumentProcessor
from utils.tokenizer import encode_history

dotenv_path = '.env'


class Chatbot:
    _user_header_tag: str = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    _assistant_header_tag: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    _prompt_template: str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" \
                            "You are an intelligent assistant." \
                            "You always provide well-reasoned answers that are both correct and helpful.\n" \
                            "The above history is a conversation between you and a human(if there isn't anything that means a new start ).\n" \
                            "each provided context have id at the start.\n" \
                            "Instructions:\n" \
                            "- **Provide** the IDs of the contexts you use to answer the user in this format: `<id1>, <id2>, ...`. if no context used put <0> in it. \n" \
                            "- The response should strictly follow this structure: `<id>` \n `answer`." \
                            "- Provide only the answer; avoid unnecessary talk or explanations.\n" \
                            "- Provide an accurate and thoughtful answer based on the context if the question is related.\n" \
                            "- If the question is unrelated or general (like greetings), respond appropriately but without referencing the context.\n" \
                            "- If you don't know the answer, simply say I don't know.\n" \
                            "Contexts:\n" \
                            "{context}\n" \
                            "{history}\n" \
                            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n" \
                            "{question}\n" \
                            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    _env_values: OrderedDict = dotenv_values(dotenv_path)

    _documentProcessor: DocumentProcessor = DocumentProcessor(chunk_size=int(_env_values["chunk_size"]),
                                                              chunk_overlap=int(_env_values["chunk_overlap"]))

    # Use nomic-embed-text to utilize all models we use from lm-studio
    _embedding: OpenAIEmbeddings = OpenAIEmbeddings(model="nomic-ai/nomic-embed-text-v1.5-GGUF",
                                                    base_url=_env_values["openAI_base_url"],
                                                    api_key=_env_values["openAI_api_key"])
    # Do not check the token length of inputs and automatically split inputs
    # longer than embedding_ctx_length. (Won't work with nomic-embed-text)
    _embedding.check_embedding_ctx_length = False
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
        self._history = None
        self.prompt_template = prompt_template
        self.limit = limit
        self._rag_prompt: PromptTemplate = PromptTemplate.from_template(prompt_template)
        self._retriever = self.__class__._milvus.as_retriever(search_type="similarity", search_kwargs={"k": limit})

        self._rag_chain = {"context": self._retriever | self._format_doc, "history": RunnableLambda(self.get_history),
                           "question": RunnablePassthrough()} | self._rag_prompt | self.__class__._llm | StrOutputParser()

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"prompt_template={self.prompt_template}, "
                f"limit={self.limit})")

    def get_response(self, query: str, history: List[Dict[str, str]], stream: bool = False) -> Iterator[str] | str:
        """
        Get response from LLM model.

        This method using the chain that we create it constructor calls the model and return the answer.

        Parameters:
        query (str): user question without embeddings.
        history (str): that chat history between user and model.
        stream (bool): if true return streamed version of answer

        Returns:
        str: output of chain invoke
        """

        self._history = encode_history(
            user_header_tag=self.__class__._user_header_tag,
            assistant_header_tag=self.__class__._assistant_header_tag,
            histories=history,
        )

        if stream:
            return self._rag_chain.stream(query)

        return self._rag_chain.invoke(query)

    def save_pdf(self, file) -> None:
        """
        Save embedded chunks into Milvus db.

        This method get PDF file and split it using DocumentProcessor class and convert them into vectors
        and save it into Milvus db.

        Parameters:
        file (file | streamlit file_uploader like objects): returning object of streamlit.file_uploader

        Returns:
        None
        """

        chunks: List[str] = self.__class__._documentProcessor.load_pdf(file=file)
        documents: List[Document] = [Document(page_content=chunk, metadate={"file_id": file.file_id}) for chunk in
                                     chunks]
        document_ids: List[str] = [str(uuid4()) for _ in documents]
        self.__class__._milvus.add_documents(documents=documents, ids=document_ids)

    def delete_pdf(self, file_id: str):
        """
        Delete vectors from a pdf file from Milvus db.

        This method deletes every vector from chunks of a specific PDF from milvus using their file_id.

        Parameters:
        file_id (str): the file_id that streamlit provide to each uploaded file.

        Returns:
        None
        """
        documents_id: List[str] = self.__class__._milvus.get_pks(expr=f"file_id == '{file_id}'")
        self.__class__._milvus.delete(ids=documents_id)

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

    def get_history(self, _):
        return self._history
