from document_processor import DocumentProcessor
from vectorizer import Vectorizer
from milvus_handler import MilvusHandler
from chatbot import Chatbot
from chat_interface import ChatInterface
from dotenv import load_dotenv, set_key, dotenv_values
from collections import OrderedDict
from typing import List, Any

dotenv_path = ".env"

def main():
    env_values = dotenv_values(dotenv_path)

    document_processor = DocumentProcessor(chunk_size=int(env_values['chunk_size']),
                                           chunk_overlap=int(env_values['chunk_overlap']))
    vectorizer = Vectorizer(model_name=env_values['embedding_model_name'])
    milvus_handler = MilvusHandler(collection_name=env_values['collection_name'],
                                   dimensions=vectorizer.dimension,
                                   milvus_uri=env_values['milvus_uri'])
    chatbot = Chatbot(openAI_base_url=env_values['openAI_base_url'],
                      openAI_api_key=env_values['openAI_base_url'],
                      model_name=env_values['LLM_model_name'])
    chat_interface = ChatInterface(document_processor, vectorizer, milvus_handler, chatbot)
    chat_interface.run()


def setup_env():
    print("Suppose this is your first time running the app!")
    print("Please answer the above question for configuration :")

    #DocumentProcessor params part
    chunk_size = size if (size := input("Enter chunk size for text split function (Enter for 256) : ")) else "256"
    set_key(dotenv_path , 'chunk_size' , chunk_size)
    chunk_overlap = overlap if (overlap := input("Enter chunk overlap for text split function (Enter for 64) : ")) else "64"
    set_key(dotenv_path , 'chunk_overlap' , chunk_overlap)

    #Vectorizer param part
    embedding_model_name = model if (model := input("Enter model name for word embedding"
        " (Enter for sentence-transformers/all-MiniLM-L6-v2) : "))\
        else "sentence-transformers/all-MiniLM-L6-v2"
    set_key(dotenv_path , 'embedding_model_name' , embedding_model_name)

    #MilvusHandler params part
    collection_name = name if (name := input("Enter collection name for Milvus db (Enter for Test) : ")) else "Test"
    set_key(dotenv_path , 'collection_name' , collection_name)
    milvus_uri = uri if (uri := input("Enter your milvus uri (Enter for http://localhost:19530) : "))\
        else "http://localhost:19530"
    set_key(dotenv_path , 'milvus_uri' , milvus_uri)

    #Chatbot params part
    openAI_base_url = url if (url := input("Enter open ai url to connect (Enter for http://localhost:1234/v1) : "))\
        else "http://localhost:1234/v1"
    set_key(dotenv_path , 'openAI_base_url' , openAI_base_url)
    openAI_api_key = key if (key := input("Enter your open ai api key (Enter for lm-studio) : ")) else "lm-studio"
    set_key(dotenv_path , 'openAI_api_key' , openAI_api_key)
    LLM_model_name = name if (name := input("Enter LLM model name "
        "(Enter for lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF) : "))\
        else "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
    set_key(dotenv_path , 'LLM_model_name' , LLM_model_name)



def update_env():
    print("Updating Configuration ... ")
    env_values: OrderedDict = dotenv_values(dotenv_path)

    for key in env_values:
        is_edit: bool = True if input(f"Value for {key} is {env_values[key]} Want to edit it ? (y/N)") else False

        if is_edit:
            updated_value: str = input(f"Enter updated value for {env_values[key]} : ")
            set_key(dotenv_path, key, updated_value)


def check_env() -> list[Any]:
    print("Start to checking parameters ...")
    env_values: OrderedDict = dotenv_values(dotenv_path)
    errors: list = []

    chunk_size: str = env_values['chunk_size']
    chunk_overlap: str = env_values['chunk_overlap']
    try:
        if (int(chunk_overlap) > int(chunk_size)):
            errors.append(f"chunk_size is less than chunk_overlap!: \n\t{e}")

    except ValueError as e:
        errors.append(f"chunk_size or chunk_overlap is invalid : \n\t{e}")

    print("chunk_size and chunk_overlap checked!")

    embedding_model_name: str = env_values["embedding_model_name"]
    if error := Vectorizer.check_model_name(model_name=embedding_model_name):
        errors.append(error)
    print("embedding model name checked!")

    milvus_uri: str = env_values["milvus_uri"]
    if error := MilvusHandler.check_milvus_uri(milvus_uri=milvus_uri):
        errors.append(error)

    print("milvus uri checked!")

    openAI_base_url: str = env_values["openAI_base_url"]
    openAI_api_key: str = env_values["openAI_api_key"]
    LLM_model_name: str = env_values["LLM_model_name"]

    if error := Chatbot.check_chatbot_params(base_url=openAI_base_url, api_key=openAI_api_key, model_name=LLM_model_name):
        errors.append(error)

    print("open ai base url and api key with LLM model name checked!")
    return errors


if __name__ == "__main__":
    print("Welcome to Rag System project !")

    if not load_dotenv(dotenv_path):
        setup_env()
        print("Setup configuration has completed!")

    option = int(input("1) Update configuration\n2) Continue to run app\nEnter your choice : "))
    while option not in (1,2):
        print("Invalid input!")
        option = int(input("1) Update configuration\n2) Continue to run app\nEnter your choice : "))

    if option == 1:
        update_env()
        print("Updating configuration has completed!")

    errors = check_env()
    while errors:
        print("Unfortunately there are some errors with your configuration."
              " Make sure Milvus and lm-studio (if you are using it) is up : ")
        print('\n\n'.join(errors))
        update_env()
        errors = check_env()

    print("All done!")
    main()
