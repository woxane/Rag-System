from dotenv import set_key, dotenv_values
from collections import OrderedDict


class EnvManager:
    def __init__(self, dotenv_path: str = ".env"):
        self.path = dotenv_path

    def setup_env(self):
        # DocumentProcessor params part
        chunk_size = size if (size := input("Enter chunk size for text split function (Enter for 256) : ")) else "256"
        set_key(self.path, 'chunk_size', chunk_size)
        chunk_overlap = overlap if (
            overlap := input("Enter chunk overlap for text split function (Enter for 64) : ")) else "64"
        set_key(self.path, 'chunk_overlap', chunk_overlap)

        # Vectorizer param part
        embedding_model_name = model if (model := input("Enter model name for word embedding"
                                                        " (Enter for sentence-transformers/all-MiniLM-L6-v2) : ")) \
            else "sentence-transformers/all-MiniLM-L6-v2"
        set_key(self.path, 'embedding_model_name', embedding_model_name)

        # MilvusHandler params part
        collection_name = name if (name := input("Enter collection name for Milvus db (Enter for Test) : ")) else "Test"
        set_key(self.path, 'collection_name', collection_name)
        milvus_uri = uri if (uri := input("Enter your milvus uri (Enter for http://localhost:19530) : ")) \
            else "http://localhost:19530"
        set_key(self.path, 'milvus_uri', milvus_uri)

        # Chatbot params part
        openAI_base_url = url if (url := input("Enter open ai url to connect (Enter for http://localhost:1234/v1) : ")) \
            else "http://localhost:1234/v1"
        set_key(self.path, 'openAI_base_url', openAI_base_url)
        openAI_api_key = key if (key := input("Enter your open ai api key (Enter for lm-studio) : ")) else "lm-studio"
        set_key(self.path, 'openAI_api_key', openAI_api_key)
        LLM_model_name = name if (name := input("Enter LLM model name "
                                                "(Enter for lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF) : ")) \
            else "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
        set_key(self.path, 'LLM_model_name', LLM_model_name)

    def update_env(self) -> None:
        """
        Update environment variables.

        This method iterate on variables that is in the dotenv that specified and if there is anything that user
        wants to change it, it will update it.

        Parameter:
        None

        Returns:
        None
        """
        env_values: OrderedDict = dotenv_values(self.path)

        for key in env_values:
            is_edit: bool = True if input(f"Value for {key} is {env_values[key]} Want to edit it ? (y/N)") else False

            if is_edit:
                updated_value: str = input(f"Enter updated value for {env_values[key]} : ")
                set_key(self.path, key, updated_value)
