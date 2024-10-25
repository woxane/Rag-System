import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any
from PIL import Image
import io


class DocumentProcessor:
    _separators: List[str] = [".", ","]

    def __init__(self, chunk_size: int = 400):
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
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        pdf = {"chunks": [], "images": []}

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

            images = page.get_images(full=True)

            for image in images:
                xref = image[0]
                image_data = pdf_document.extract_image(xref)
                image_bytes = image_data["image"]

                pil_image = Image.open(io.BytesIO(image_bytes))
                pdf['images'].append(pil_image)

        pdf_document.close()

        text = text.replace("\n", " ")
        pdf['chunks'] = self.text_splitter.split_text(text)
        return pdf 
