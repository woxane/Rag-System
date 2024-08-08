import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    _separators = [
                "\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001",
                "\uff0e", "\u3002", ""
    ]

    def __init__(self, chunk_size=256, chunk_overlap=64):
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=self._separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_pdf(self, uploaded_file):
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()

        chunks = self.text_splitter.split_text(text)
        return chunks
