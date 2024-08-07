from document_processor import DocumentProcessor
from vectorizer import Vectorizer
from milvus_handler import MilvusHandler
from chatbot import Chatbot
from chat_interface import ChatInterface
from dotenv import load_dotenv, set_key, dotenv_values


dotenv_path = ".env"

def main():
    document_processor = DocumentProcessor()
    vectorizer = Vectorizer()
    milvus_handler = MilvusHandler("RAG_test", 384)
    chatbot = Chatbot("http://localhost:1234/v1", "lm-studio", 'lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF')
    chat_interface = ChatInterface(document_processor, vectorizer, milvus_handler, chatbot)
    chat_interface.run()


def setup_env():
    #DocumentProcessor params part
    chunk_size = size if (size := input("Enter chunk size for text split function (Enter for 256) : ")) else 256
    set_key(dotenv_path , 'chunk_size' , chunk_size)
    chunk_overlap = overlap if (overlap := input("Enter chunk overlap for text split function (Enter for 64) : ")) else 64
    set_key(dotenv_path , 'chunk_overlap' , chunk_overlap)

    #Vectorizer param part
    embedding_model_name = model if (model := input("Enter model name for word embedding (Enter for sentence-transformers/all-MiniLM-L6-v2) : ")) else "sentence-transformers/all-MiniLM-L6-v2"
    set_key(dotenv_path , 'embedding_model_name' , embedding_model_name)

    #MilvusHandler params part
    collection_name = input("Enter collection name for Milvus db : ")
    set_key(dotenv_path , 'collection_name' , collection_name)
    dimension = input("Enter number of dimension for your db (it must match with the word embedding model) : ")
    set_key(dotenv_path , 'dimension' , dimension)
    milvus_uri = input("Enter your milvus uri : ")
    set_key(dotenv_path , 'milvus_uri' , milvus_uri)

    #Chatbot params part
    openAI_base_url = input("Enter open ai url to connect : ")
    set_key(dotenv_path , 'openAI_base_url' , openAI_base_url)
    openAI_api_key = input("Enter your open ai api key : ")
    set_key(dotenv_path , 'openAI_api_key' , openAI_api_key)
    LLM_model_name = input("Enter LLM model name : ")
    set_key(dotenv_path , 'LLM_model_name' , LLM_model_name)


if __name__ == "__main__":
    load_dotenv(dotenv_path)
    main()
