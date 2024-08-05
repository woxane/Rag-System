from document_processor import DocumentProcessor
from vectorizer import Vectorizer
from milvus_handler import MilvusHandler
from chatbot import Chatbot
from chat_interface import ChatInterface

def main():
    document_processor = DocumentProcessor()
    vectorizer = Vectorizer()
    milvus_handler = MilvusHandler("RAG_test", 384)
    chatbot = Chatbot("http://localhost:1234/v1", "lm-studio", 'lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF')
    chat_interface = ChatInterface(document_processor, vectorizer, milvus_handler, chatbot)
    chat_interface.run()

if __name__ == "__main__":
    main()
