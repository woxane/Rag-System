from document_processor import DocumentProcessor
from vectorizer import Vectorizer
from milvus_handler import MilvusHandler
from chatbot import Chatbot
from chat_interface import ChatInterface
from dotenv import load_dotenv, set_key, dotenv_values
from collections import OrderedDict

dotenv_path = ".env"

def main():
    env_values = dotenv_values(dotenv_path)

    document_processor = DocumentProcessor(chunk_size=env_values['chunk_size'], chunk_overlap=env_values['chunk_overlap'])
    vectorizer = Vectorizer(model_name=env_values['embedding_model_name'])
    milvus_handler = MilvusHandler(collection_name=env_values['collection_name'], dimensions=env_values["dimension"], milvus_uri=env_values['milvus_uri'])
    chatbot = Chatbot(openAI_base_url=env_values['openAI_base_url'],
                      openAI_api_key=env_values['openAI_base_url'],
                      model_name=env_values['LLM_model_name'])
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



def update_env():
    env_values: OrderedDict = dotenv_values(dotenv_path)

    for key in env_values:
        is_edit: bool = True if input(f"Value for {key} is {env_values[key]} Want to edit it ? (y/N)") else False

        if is_edit:
            updated_value: str = input(f"Enter updated value for {env_values[key]} : ")
            set_key(dotenv_path, key, updated_value)


if __name__ == "__main__":
    if not load_dotenv(dotenv_path):
        setup_env()

    main()
