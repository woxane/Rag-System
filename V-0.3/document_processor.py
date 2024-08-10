import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


class DocumentProcessor:
    _separators: List[str] = [
        "\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001",
        "\uff0e", "\u3002", ""
    ]

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            separators=self._separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )


    def __repr__(self):
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size!r}, chunk_overlap={self.chunk_overlap!r})"

    def load_pdf(self, file) -> List[Document]:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()

        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata={"file_id": file.file_id}) for chunk in chunks]
        return documents
