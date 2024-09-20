from dotenv import load_dotenv, set_key, dotenv_values
from collections import OrderedDict
from os import system
import atexit
from sys import path

path.append('../')

from utils.document_processor import DocumentProcessor

dotenv_path = ".env"

def main():
    # Run the Streamlit app interface
    system("streamlit run chat_interface.py")

def setup_env():
    # Function to set up environment variables interactively
    print("Suppose this is your first time running the app!")
    print("Please answer the above question for configuration:")

    # DocumentProcessor parameters
    chunk_size = size if (size := input("Enter chunk size for text split function (Enter for 256): ")) else "256"
    set_key(dotenv_path, 'chunk_size', chunk_size)
    chunk_overlap = overlap if (overlap := input("Enter chunk overlap for text split function (Enter for 64): ")) else "64"
    set_key(dotenv_path, 'chunk_overlap', chunk_overlap)

    # Vectorizer parameters
    embedding_model_name = model if (model := input("Enter model name for word embedding (Enter for sentence-transformers/all-MiniLM-L6-v2): ")) \
        else "sentence-transformers/all-MiniLM-L6-v2"
    set_key(dotenv_path, 'embedding_model_name', embedding_model_name)

    # MilvusHandler parameters
    collection_name = name if (name := input("Enter collection name for Milvus db (Enter for Test): ")) else "Test"
    set_key(dotenv_path, 'collection_name', collection_name)
    milvus_uri = uri if (uri := input("Enter your milvus uri (Enter for http://localhost:19530): ")) \
        else "http://localhost:19530"
    set_key(dotenv_path, 'milvus_uri', milvus_uri)

    # Chatbot parameters
    openAI_base_url = url if (url := input("Enter open ai url to connect (Enter for http://localhost:1234/v1): ")) \
        else "http://localhost:1234/v1"
    set_key(dotenv_path, 'openAI_base_url', openAI_base_url)
    openAI_api_key = key if (key := input("Enter your open ai api key (Enter for lm-studio): ")) else "lm-studio"
    set_key(dotenv_path, 'openAI_api_key', openAI_api_key)
    LLM_model_name = name if (name := input("Enter LLM model name (Enter for lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF): ")) \
        else "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
    set_key(dotenv_path, 'LLM_model_name', LLM_model_name)

def update_env():
    # Function to update existing environment variables
    print("Updating Configuration...")
    env_values: OrderedDict = dotenv_values(dotenv_path)

    for key in env_values:
        # Ask user if they want to edit the existing value
        is_edit: bool = True if input(f"Value for {key} is {env_values[key]}. Want to edit it? (y/N): ") else False

        if is_edit:
            updated_value: str = input(f"Enter updated value for {env_values[key]} : ")
            set_key(dotenv_path, key, updated_value)


if __name__ == "__main__":
    # Register cleanup function to be called on exit
    atexit.register(DocumentProcessor.data_clean_up)

    print("Welcome to Rag System project!")

    # Load environment variables; setup if not found
    if not load_dotenv(dotenv_path):
        setup_env()
        print("Setup configuration has completed!")

    option = int(input("1) Update configuration\n2) Continue to run app\nEnter your choice: "))
    while option not in (1, 2):
        print("Invalid input!")  # Handle invalid input
        option = int(input("1) Update configuration\n2) Continue to run app\nEnter your choice: "))

    if option == 1:
        update_env()
        print("Updating configuration has completed!")

    print("All done!")
    main()
