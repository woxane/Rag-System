import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any
from PIL import Image
import io
import base64
import os
import glob


class DocumentProcessor:
    _separators: List[str] = [".", ","]
    base_directory = ".data/"

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
        Extract text and images from a PDF file and split the text into chunks.

        This method opens a PDF file, extracts its text and images, and processes them. It uses the `RecursiveCharacterTextSplitter` from LangChain to divide the text into manageable chunks.
        The images are converted to base64 format and saved to a specified directory, while the extracted tables are also organized for further use.

        Parameters:
        -----------
        file : file or streamlit file_uploader-like object
            A file object returned by streamlit's file_uploader or similar objects, representing the PDF to be processed.

        Returns:
        --------
        Dict[str, List[Any]]
            A dictionary containing:
                - 'chunks': A list of text chunks obtained by splitting the PDF text.
                - 'images': A list of tuples containing the file path, base64-encoded image data, and metadata for each extracted image.
                - 'tables': A list of tuples with extracted table data and their associated page and table numbers.
        """
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        pdf = {"chunks": [], "images": [], "tables": []}

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            tables_datas = self.extract_tables(page)

            for table_num, table in enumerate(tables_datas):
                table_markdown = self.convert_table_to_markdown(table['table'].extract())
                full_tabel_data = '\n' + table['above_text'] + '\n' + table_markdown + '\n' + table['below_text']
                pdf['tables'].append((full_tabel_data, page_num, table_num))

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

    def delete_images(self, file_id: str) -> List[str]:
        """
        Delete images associated with a specific file ID.

        This method searches for and removes image files from the filesystem that match the given file ID.
        It uses a pattern to identify files and attempts to delete each one, reporting any errors encountered during the deletion process.

        Parameters:
        -----------
        file_id : str
            The identifier associated with the images to be deleted. This ID is used to match files in the specified directory.

        Returns:
        --------
        List[str]
            A list of file paths for the images that were attempted to be deleted. This list may include files that could not be removed due to errors.
        """
        pattern = os.path.join(self.__class__.base_directory, f'*{file_id}*')

        files_to_delete = glob.glob(pattern)

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error: {file_path} : {e.strerror}")

        return files_to_delete

    def extract_tables(self, page) -> List[Dict[str, Any]]:
        """
        Extract tables and their contextual text from a PDF page.

        This method identifies tables within a specified page of a PDF document and retrieves the text located above and below each table.
        It applies redaction annotations to the tables and organizes the extracted information into a structured format for further processing.

        Parameters:
        -----------
        page : Page
            The page object from which tables and contextual text are to be extracted.

        Returns:
        --------
        List[Dict[str, Any]]
            A list of dictionaries, each containing:
                - 'above_text': The text located immediately above the table.
                - 'table': The table object extracted from the page.
                - 'below_text': The text located immediately below the table.
        """

        text_data = page.get_text("dict")
        tables = page.find_tables()
        results = []

        for table in tables:
            table_bbox = table.bbox
            page.add_redact_annot(table_bbox)
            page.apply_redactions()

            # Store the closest lines
            above_text = ""
            below_text = ""

            closest_above_y = float('-inf')
            closest_below_y = float('inf')

            for block in text_data["blocks"]:
                # if the block type is text (0: text, 1: image)
                if block['type'] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_bbox = span["bbox"]

                            if table_bbox[1] > text_bbox[3] > closest_above_y:
                                above_text = span["text"]
                                closest_above_y = text_bbox[3]

                            elif table_bbox[3] < text_bbox[1] < closest_below_y:
                                below_text = span["text"]
                                closest_below_y = text_bbox[1]

            table_with_context = {
                "above_text": above_text.strip(),
                "table": table,
                "below_text": below_text.strip()
            }
            results.append(table_with_context)

        return results

    def convert_table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Convert a table to Markdown format.

        This method takes a two-dimensional list representing a table and converts it into a Markdown-formatted string.
        It creates a separator for the columns and formats the rows appropriately.

        Parameters:
        -----------
        table : List[List[str]]
            A list of lists, where each inner list represents a row of the table and contains string values for each column.

        Returns:
        --------
        str
            A string representation of the table in Markdown format.
        """
        number_of_columns = len(table[0])

        separator = "|" + "|".join(["---"] * number_of_columns) + "|"
        markdown_rows = ["|" + "|".join(row) + "|" for row in table]
        markdown_table = "\n".join([markdown_rows[0], separator] + markdown_rows[1:])

        return markdown_table

    @classmethod
    def data_clean_up(cls) -> None:
        """
        Clean up the directory by deleting all files matching the specified pattern.

        This class method searches for and removes all files in the base directory.
        It attempts to delete each file and reports any errors encountered during the deletion process.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        pattern = os.path.join(cls.base_directory, '*')

        files_to_delete = glob.glob(pattern)

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error: {file_path} : {e.strerror}")
