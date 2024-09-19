from sys import path
from dotenv import dotenv_values
from collections import OrderedDict
from typing import List, Iterator, Dict, Tuple
from uuid import uuid4, UUID
from pymilvus import MilvusClient
from openai import OpenAI as lm_studio
from PIL import Image
import base64
from io import BytesIO

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
                            "The above history is a conversation between you and a human (if there isn't anything, it means a new start).\n" \
                            "Instructions:\n" \
                            "- Detect the language of the user's question and respond in the same language, even if the context is in a different language.\n" \
                            "- Provide only the answer; avoid unnecessary talk or explanations.\n" \
                            "- Provide an accurate and thoughtful answer based on the context, whether it is related to an image analysis or a text-based context.\n" \
                            "- If the question is unrelated or general (like greetings), respond appropriately but without referencing the context.\n" \
                            "- If you don't know the answer, simply say, 'I don't know.'\n" \
                            "Contexts:\n" \
                            "{context}\n" \
                            "{history}\n" \
                            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n" \
                            "{question}\n" \
                            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    _table_analyzation_prompt_template: str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" \
                                              "You are an intelligent assistant with the ability to analyze and summarize data from tables.\n" \
                                              "Instructions:\n" \
                                              "- Look at the table provided below.\n" \
                                              "- Provide a detailed summary that includes key insights, trends, and any notable figures from the table.\n" \
                                              "- Ensure that your summary includes the actual data from the table where relevant.\n" \
                                              "- The summary should be comprehensive, capturing all points and key numbers from the table.\n" \
                                              "Table:\n" \
                                              "{table_data}\n" \
                                              "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    _env_values: OrderedDict = dotenv_values(dotenv_path)

    _documentProcessor: DocumentProcessor = DocumentProcessor(chunk_size=int(_env_values["chunk_size"]))

    _embedding_model_name = "Alibaba-NLP/gte-multilingual-base"
    _embedding_model_kwargs = {"trust_remote_code": True}
    _embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=_embedding_model_name,
                                                              model_kwargs=_embedding_model_kwargs)

    # Do not check the token length of inputs and automatically split inputs
    # longer than embedding_ctx_length. (Won't work with nomic-embed-text)
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
        self._used_contexts = []
        self.prompt_template = prompt_template
        self.limit = limit
        self._rag_prompt: PromptTemplate = PromptTemplate.from_template(prompt_template)
        self._table_analyze_prompt: PromptTemplate = PromptTemplate.from_template(self.__class__._table_analyzation_prompt_template)
        self._retriever = self.__class__._milvus.as_retriever(search_type="similarity", search_kwargs={"k": limit})

        self._rag_chain = {"context": self._retriever | self._format_doc, "history": RunnableLambda(self.get_history),
                           "question": RunnablePassthrough()} | self._rag_prompt | self.__class__._llm | StrOutputParser()

        self._table_analyze_chain = self._table_analyze_prompt | self.__class__._llm | StrOutputParser()

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

        chain = self._rag_chain

        if stream:
            return chain.stream(query)

        return chain.invoke(query)

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
        # TODO: change the way of saving file path
        docs = []

        pdf_data = self.__class__._documentProcessor.load_pdf(file=file)
        chunks = pdf_data['chunks']
        images = pdf_data['images']
        tables = pdf_data['tables']

        images_analyzation = [(self.analyze_image(image), file_path, image_info) for file_path, image, image_info in images]
        tables_analyzation = [(self.analyze_table(full_table_data), full_table_data, page_num, table_num) for full_table_data, page_num, table_num in tables]

        documents = []

        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk[0],
                    metadata={
                        "file_id": file.file_id,
                        "file_name": file.name,
                        "chunk_number": idx + 1,
                        "data_type": "text",
                        "file_path": "",
                        "page_num": "",
                        "image_num": "",
                        "table_num": "",
                        "table_markdown": "",
                    }
                )
            )

        for idx, (analyze, file_path, image_info) in enumerate(images_analyzation, len(chunks)):
            documents.append(
                Document(
                    page_content=analyze,
                    metadata={
                        "file_id": file.file_id,
                        "file_name": file.name,
                        "chunk_number": idx + 1,
                        "data_type": "image-analyze",
                        "file_path": file_path,
                        "page_num": str(image_info['page_num']),
                        "image_num": str(image_info['image_num']),
                        "table_num": "",
                        "table_markdown": "",
                    }
                )
            )

        for idx, (analyze, full_table_data,  page_num, table_num) in enumerate(tables_analyzation, len(chunks) + len(images_analyzation)):
            documents.append(
                Document(
                    page_content=analyze,
                    metadata={
                        "file_id": file.file_id,
                        "file_name": file.name,
                        "chunk_number": idx + 1,
                        "data_type": "table-analyze",
                        "file_path": "",
                        "page_num": str(page_num),
                        "image_num": "",
                        "table_num": str(table_num),
                        "table_markdown": full_table_data,
                    }
                )
            )

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
        deleted_images: List[str] = self.__class__._documentProcessor.delete_images(file_id=file_id)
        self.__class__._milvus.delete(ids=documents_id)

    def get_formatted_references(self) -> List[str]:
        """
        Get texts near the real reference by given chunk_number.

        This method Filters and sorts the chunks based on their proximity to the given chunk_number.

        Parameters:
        file_id (list): The file id of the reference text.
        chunk_number (int): The reference chunk number to filter around.

        Returns:
            list: Sorted list of chunks near the specified chunk_number.
        """

        documents = self._used_contexts

        references = []
        if not documents:
            return []

        for document in documents:
            if document.metadata['data_type'] == 'text':
                file_datas = self.__class__._pymilvus_client.query(
                    collection_name=self.__class__._env_values['collection_name'],
                    filter=f"file_id == '{document.metadata['file_id']}'"
                )

                chunk_number = document.metadata['chunk_number']

                near_references = sorted(
                    filter(lambda file_data:
                           chunk_number - 1 <= int(file_data['chunk_number']) <= chunk_number + 1, file_datas),
                    key=lambda data: data['chunk_number']
                )

                near_references = sorted(near_references, key=lambda data: data['chunk_number'])

                reference_index = next((i for i, chunk in enumerate(near_references) if int(chunk['chunk_number']) == chunk_number), None)
                chunk_texts = list(map(lambda chunk: chunk['text'], near_references))

                if reference_index != None:
                    chunk_texts[reference_index] = "<mark style='background-color: yellow'>" + chunk_texts[reference_index] + "</mark>"

                references.append(" ".join(chunk_texts))

            elif document.metadata['data_type'] == 'image-analyze':
                try:
                    file_path = document.metadata['file_path']
                    image = Image.open(file_path)

                    buffered = BytesIO()
                    image.save(buffered, format=image.format)

                    mime_type = f"image/{image.format.lower()}"

                    image_b64 = base64.b64encode(buffered.getvalue()).decode()

                    image_tag = f'<img src="data:{mime_type};base64,{image_b64}" alt="alt text">'
                    information_tag = f"<p>This image located in {document.metadata['file_name']} at page number {document.metadata['page_num']} </p>"

                    references.append(image_tag + '\n' + information_tag)

                except Exception as e:
                    print(e)


            elif document.metadata['data_type'] == 'table-analyze':
                table_markdown = document.metadata['table_markdown']
                table_information = f"<p>This table located in {document.metadata['file_name']} at page number {document.metadata['page_num']} </p>"

                references.append(table_markdown + '\n' + table_information)

        return references

    def analyze_image(self, image_base64: str, response_language: str = "Persian") -> str:
        llava_model_name = "xtuner/llava-llama-3-8b-v1_1-gguf"
        client: lm_studio = lm_studio(base_url="http://localhost:1234/v1", api_key="lm-studio")

        analyze_prompt = "Instructions:\n" \
                         "- **List** all features in the image.\n" \
                         "- If there is any text or table in the image describe a summary of it."

        completion = client.chat.completions.create(
            model=llava_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "This is a chat between a user and an assistant. The assistant is helping the user to describe an image.\n" + analyze_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "analyze this image for me."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
            stream=False
        )

        return completion.choices[0].message.content


    def analyze_table(self, table_markdown):
        return self._table_analyze_chain.invoke(table_markdown)



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
        self._used_contexts = list(docs)

        formated_documents = "\n".join(doc.page_content for doc in docs)

        return formated_documents

    def get_history(self, _):
        return self._history

    def get_latest_context(self):
        return self._used_contexts
