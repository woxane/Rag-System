from sys import path
from dotenv import dotenv_values
from collections import OrderedDict
from typing import List, Iterator, Dict, Tuple
from uuid import uuid4, UUID
from pymilvus import MilvusClient

from langchain_milvus import Milvus
from langchain_openai import OpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
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
                            "Each provided context have an their own ID at the start like this: <ID>context</ID>\n" \
                            "Instructions:\n" \
                            "- **List** the tagged IDs of the contexts you use to answer the user at the start of your response with this format: `<id> <id> ...`. If no context is used, put `<0>::`. \n" \
                            "- **Do not** include any text before the context IDs. \n" \
                            "- **Separate** the context IDs from your answer using `::`.\n" \
                            "- Provide only the answer; avoid unnecessary talk or explanations.\n" \
                            "- Provide an accurate and thoughtful answer based on the context if the question is related.\n" \
                            "- If the question is unrelated or general (like greetings), respond appropriately but without referencing the context.\n" \
                            "- If you don't know the answer, simply say, I don't know.\n" \
                            "Contexts:\n" \
                            "{context}\n" \
                            "{history}\n" \
                            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n" \
                            "{question}\n" \
                            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    _env_values: OrderedDict = dotenv_values(dotenv_path)

    _documentProcessor: DocumentProcessor = DocumentProcessor(chunk_size=int(_env_values["chunk_size"]))

    _embedding_model_name = "Alibaba-NLP/gte-multilingual-base"
    _embedding_model_kwargs = {"trust_remote_code": True}
    _embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=_embedding_model_name,
                                                              model_kwargs=_embedding_model_kwargs)

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

    _pymilvus_client: MilvusClient = MilvusClient(
        uri=_env_values["milvus_uri"]
    )

    def __init__(self, prompt_template: str = _prompt_template, limit: int = 3):
        self._history = None
        self._latest_contexts = None
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

        chunks: List[str] = self.__class__._documentProcessor.load_pdf(file=file)["chunks"]
        documents: List[Document] = [Document(
            page_content=chunks[chunk_number],
            metadata={"file_id": file.file_id, "file_name": file.name, "chunk_number": chunk_number + 1}
        ) for chunk_number in range(len(chunks))]
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

    def get_formatted_references(self, chunk_number: int, file_id: str) -> str:
        """
        Get texts near the real reference by given chunk_number.

        This method Filters and sorts the chunks based on their proximity to the given chunk_number.

        Parameters:
        file_id (list): The file id of the reference text.
        chunk_number (int): The reference chunk number to filter around.

        Returns:
            list: Sorted list of chunks near the specified chunk_number.
        """
        file_datas = self.__class__._pymilvus_client.query(
            collection_name=self.__class__._env_values['collection_name'],
            filter=f"file_id == '{file_id}'"
        )

        near_references = sorted(
            filter(lambda file_data:
                    chunk_number - 1 <= int(file_data['chunk_number']) <= chunk_number + 1, file_datas),
                    key=lambda data: data['chunk_number']
        )

        near_references = sorted(near_references, key=lambda data: data['chunk_number'])

        reference_index = next((i for i, chunk in enumerate(near_references) if int(chunk['chunk_number']) == chunk_number), None)
        chunk_texts = list(map(lambda chunk: chunk['text'], near_references))

        if reference_index:
            chunk_texts[reference_index] = "<mark style='background-color: yellow'>" + chunk_texts[reference_index] + "</mark>"
        
        return " ".join(chunk_texts)


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

    def _format_doc(self, docs: List[Document]) -> str:
        """
        Joins page_content of each element using \n\n.

        This method get searched documents and a tag with a specifc ID as a chunk number to each one of them.

        Parameters:
        docs (List[Documents]): List of searched documents.

        Returns:
        str: output of joins on the page contents.
        """
        formated_contexts = [(doc.metadata['file_id'], "<" + str(doc.metadata['chunk_number']) + ">" + doc.page_content +
                            "</" + str(doc.metadata['chunk_number']) + ">") for doc in docs]

        self._latest_context = formated_contexts

        formated_documents = "\n".join("<" + str(doc.metadata['chunk_number']) + ">" + doc.page_content +
                            "</" + str(doc.metadata['chunk_number']) + ">" for doc in docs)

        return formated_documents

    def get_history(self, _):
        return self._history
    
    def get_latest_context(self):
        return self._latest_context
