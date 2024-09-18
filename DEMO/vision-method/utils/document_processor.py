import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any
from PIL import Image
import io
import base64
import os


class DocumentProcessor:
    _separators: List[str] = [".", ","]

    def __init__(self, chunk_size: int = 400):
        self.base_directory = ".data/"
        self.chunk_size = chunk_size
        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            separators=self._separators,
            chunk_size=chunk_size,
            length_function=len,
            is_separator_regex=False,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size!r})"

    def load_pdf(self, file) -> Dict[str, List[Any]]:
        """
        Extracts text and images from a PDF file and splits it into chunks (just texts)

        This method using split the text using langchain.text_splitter.RecursiveCharacterTextSplitter to split the text.

        Parameters:
        file (file | streamlit file_uploader like objects): returning object of streamlit.file_uploader

        Returns:
        Dict[str, List[Any]] = chunks: chunks that separated using RecursiveCharacterTextSplitter
                               images: images that finded in the pdf file.
        """
        # TODO: change the way of saving file path
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        pdf = {"chunks": [], "images": []}

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

            images = page.get_images(full=True)

            for image_index in range(len(images)):
                image = images[image_index]

                xref = image[0]
                image_data = pdf_document.extract_image(xref)
                image_bytes = image_data["image"]
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                image = Image.open(io.BytesIO(image_bytes))
                image_format = image.format

                file_path = f"{self.base_directory}{file.file_id}_{page_num}_{image_index}.{image_format.lower()}"

                image_info = {
                    "page_num": page_num,
                    "image_num": image_index
                }

                directory = os.path.dirname(file_path)

                if not os.path.exists(directory):
                    os.makedirs(directory)

                image.save(file_path)

                pdf['images'].append((file_path, image_b64, image_info))

        pdf_document.close()

        text = text.replace("\n", " ")

        chunks = list(map(lambda chunk: (chunk, "None"), self.text_splitter.split_text(text)))
        pdf['chunks'] = chunks
        return pdf
