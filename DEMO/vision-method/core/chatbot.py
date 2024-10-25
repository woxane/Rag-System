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
    # User and assistant header tags for conversation context
    _user_header_tag: str = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    _assistant_header_tag: str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # Template for the prompt sent to the assistant, including system instructions
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
    # Template for analyzing tables, with instructions for the assistant
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
    # Load environment variables from a .env file
    _env_values: OrderedDict = dotenv_values(dotenv_path)

    # Initialize DocumentProcessor with a specified chunk size
    _documentProcessor: DocumentProcessor = DocumentProcessor(chunk_size=int(_env_values["chunk_size"]))

    # Specify the embedding model and its parameters
    _embedding_model_name = "Alibaba-NLP/gte-multilingual-base"
    _embedding_model_kwargs = {"trust_remote_code": True}
    _embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=_embedding_model_name,
                                                              model_kwargs=_embedding_model_kwargs)

    # Initialize the OpenAI model for generating responses
    _llm: OpenAI = OpenAI(base_url=_env_values["openAI_base_url"],
                          api_key=_env_values["openAI_api_key"],
                          model=_env_values["LLM_model_name"])

    # Configure the Milvus database connection for storing embeddings
    _milvus: Milvus = Milvus(
        embedding_function=_embedding,
        connection_args={"uri": _env_values["milvus_uri"]},
        collection_name=_env_values["collection_name"],
        drop_old=True,
    )

    # Initialize a Milvus client for managing the database
    _pymilvus_client: MilvusClient = MilvusClient(
        uri=_env_values["milvus_uri"]
    )

    def __init__(self, prompt_template: str = _prompt_template, limit: int = 3):
        self._history = None  # Stores the history of the conversation
        self._used_contexts = []  # Keeps track of latest contexts used in the conversation
        self.prompt_template = prompt_template  # Sets the prompt template
        self.limit = limit  # Sets the maximum number of results to retrieve
        self._rag_prompt: PromptTemplate = PromptTemplate.from_template(prompt_template)  # Creates a prompt template for RAG
        self._table_analyze_prompt: PromptTemplate = PromptTemplate.from_template(
            self.__class__._table_analyzation_prompt_template)  # Creates a prompt template for table analysis
        self._retriever = self.__class__._milvus.as_retriever(search_type="similarity", search_kwargs={"k": limit})  # Configures the retriever

        # Define the RAG chain combining context retrieval, formatting, and response generation
        self._rag_chain = {"context": self._retriever | self._format_doc, "history": RunnableLambda(self.get_history),
                           "question": RunnablePassthrough()} | self._rag_prompt | self.__class__._llm | StrOutputParser()

        # Define the table analysis chain for summarizing table data
        self._table_analyze_chain = self._table_analyze_prompt | self.__class__._llm | StrOutputParser()

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"prompt_template={self.prompt_template}, "
                f"limit={self.limit})")

    def get_response(self, query: str, history: List[Dict[str, str]], stream: bool = False) -> Iterator[str] | str:
        """
        Retrieve a response from the LLM model based on the user's query.

        This method uses the predefined chain to interact with the language model, processing the user's query along with the chat history.
        It can return the response in a streamed format if specified.

        Parameters:
        -----------
        query : str
            The question posed by the user, without any embeddings.
        history : List[Dict[str, str]]
            The chat history containing previous exchanges between the user and the model.
        stream : bool, optional
            If set to True, the method returns a streamed version of the response; otherwise, it returns the complete response (default is False).

        Returns:
        --------
        Iterator[str] | str
            The output of the chain invocation, which can either be a streamed response or a single complete response.
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
        Save a PDF file and its content into the Milvus database.

        This method processes the input PDF file by extracting its text, images, and tables, converting them into vector representations
        and storing them in the Milvus database.
        It utilizes the DocumentProcessor class to split the PDF into chunks and analyze each component, including text, images, and tables.
        For each component, it creates a `Document` object with appropriate metadata and stores these objects in the Milvus database.

        Parameters:
        -----------
        file : file or streamlit file_uploader-like object
            A file object returned by streamlit's file_uploader or similar objects, representing the PDF to be processed.

        Returns:
        --------
        None
            The function doesn't return any value; it stores the extracted and processed data directly into the Milvus database.
        """

        pdf_data = self.__class__._documentProcessor.load_pdf(file=file)
        chunks = pdf_data['chunks']
        images = pdf_data['images']
        tables = pdf_data['tables']

        images_analyzation = [(self.analyze_image(image), file_path, image_info) for file_path, image, image_info in
                              images]
        tables_analyzation = [(self.analyze_table(full_table_data), full_table_data, page_num, table_num) for
                              full_table_data, page_num, table_num in tables]

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

        for idx, (analyze, full_table_data, page_num, table_num) in enumerate(tables_analyzation,
                                                                              len(chunks) + len(images_analyzation)):
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
        Delete vectors associated with a PDF file from the Milvus database.

        This method removes all vector data related to the chunks of a specific PDF file stored in the Milvus database.
        It identifies the vectors using the `file_id` and deletes the corresponding records.
        Additionally, it handles the removal of any images associated with the PDF through the DocumentProcessor class.

        Parameters:
        -----------
        file_id : str
            The unique identifier assigned to each uploaded file, used to locate and delete its corresponding vectors in the database.

        Returns:
        --------
        None
            The function doesn't return any value; it removes the relevant data from the Milvus database.
        """
        documents_id: List[str] = self.__class__._milvus.get_pks(expr=f"file_id == '{file_id}'")
        deleted_images: List[str] = self.__class__._documentProcessor.delete_images(file_id=file_id)
        self.__class__._milvus.delete(ids=documents_id)

    def get_formatted_references(self) -> List[str]:
        """
        Retrieve and format reference texts or data near a specified chunk number.

        This method filters and sorts text, image, and table chunks based on their proximity to a provided chunk number, returning them in a formatted way.
        It highlights text chunks near the reference and provides base64-encoded image and table data.
        The references are fetched from `_used_contexts` and queried from the Milvus database.

        Parameters:
        -----------
        None

        Returns:
        --------
        List[str]
            A sorted list of formatted references (texts, images, or tables) around the specified chunk number. Each reference is provided with the necessary context, such as chunk number, file name, and page number.
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

                near_references = filter(lambda file_data: file_data['data_type'] == 'text', near_references)

                near_references = sorted(near_references, key=lambda data: data['chunk_number'])

                reference_index = next(
                    (i for i, chunk in enumerate(near_references) if int(chunk['chunk_number']) == chunk_number), None)
                chunk_texts = list(map(lambda chunk: chunk['text'], near_references))

                if reference_index != None:
                    chunk_texts[reference_index] = "<mark style='background-color: yellow'>" + chunk_texts[
                        reference_index] + "</mark>"

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
                    information_tag = f"<p>This image located in <b>{document.metadata['file_name']}</b> at page number <b>{document.metadata['page_num']}</b> </p>"

                    references.append(image_tag + '\n' + information_tag)

                except Exception as e:
                    print(e)


            elif document.metadata['data_type'] == 'table-analyze':
                table_markdown = document.metadata['table_markdown']
                table_information = f"<p>This table located in <b>{document.metadata['file_name']}</b> at page number <b>{document.metadata['page_num']}</b> </p>"

                references.append(table_markdown + '\n' + table_information)

        return references

    def analyze_image(self, image_base64: str) -> str:
        """
        Analyze an image and provide a detailed description.

        This method sends an image (in base64 format) to a Vision model hosted locally, which analyzes the image and returns a description.
        The analysis includes listing features of the image and summarizing any text or table found within the image.

        Parameters:
        -----------
        image_base64 : str
            The base64-encoded representation of the image to be analyzed.

        Returns:
        --------
        str
            A detailed description of the image, including any text or table content, generated by the Vision model.
        """
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

    def analyze_table(self, table_markdown: str) -> str:
        """
        Analyze a table provided in Markdown format.

        This method processes a table in Markdown format using a pre-defined table analysis chain.
        It passes the table data through the chain and returns the result of the analysis.

        Parameters:
        -----------
        table_markdown : str
            The table content in Markdown format to be analyzed.

        Returns:
        --------
        str
            The result of the table analysis as generated by the table analysis chain.
        """
        return self._table_analyze_chain.invoke(table_markdown)

    def _format_doc(self, docs: List[Document]) -> str:
        """
        Format a list of documents by joining their content.

        This method takes a list of `Document` objects and joins the `page_content` of each document into a single string, separated by double newlines.
        It also updates the `_used_contexts` attribute with the provided documents for potential future reference.

        Parameters:
        -----------
        docs : List[Document]
            A list of searched documents containing the content to be formatted.

        Returns:
        --------
        str
            A single string containing the combined page contents of the documents, separated by double newlines.
        """
        self._used_contexts = list(docs)

        formated_documents = "\n".join(doc.page_content for doc in docs)

        return formated_documents

    def get_history(self, _):
        """
        Retrieve the history of interactions.

        This method returns the stored history of interactions that have taken place, which may include previous questions and answers.

        Parameters:
        -----------
        _ : Any
            An unused parameter, kept for consistency in method signatures.

        Returns:
        --------
        Any
            The stored history of interactions.
        """
        return self._history

    def get_latest_context(self):
        """
        Get the most recent contexts used for processing.

        This method returns the latest set of contexts (documents) that have been used for processing within the system.
        These contexts can be utilized for further interactions or analyses.

        Parameters:
        -----------
        None

        Returns:
        --------
        List[Document]
            A list of the most recently used document contexts.
        """
        return self._used_contexts
