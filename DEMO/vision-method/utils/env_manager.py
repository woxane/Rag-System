from dotenv import set_key, dotenv_values
from collections import OrderedDict
from typing import Dict, Tuple


class EnvManager:
    def __init__(self, dotenv_path: str = ".env"):
        # DRY: set environment items with why we need that and recommended value
        self._environment_items: Dict[Tuple[str, str], str] = {
            ("chunk_size", "chunk size for text splitting"): "256",
            ("chunk_overlap", "chunk overlap for text splitting"): "64",
            ("embedding_model_name", "name of the embedding model(needs to exist in hugginface)"): "sentence-transformers/all-MiniLM-L6-v2",
            ("collection_name", "collection name for Milvus db"): "Test",
            ("milvus_uri", "your milvus uri"): "http://localhost:19530",
            ("openAI_base_url", "your open ai base url for connection"): "http://localhost:1234/v1",
            ("openAI_api_key", "your open ai api key"): "lm-studio",
            ("LLM_model_name", "LLM model name"): "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        }
        
        self.path = dotenv_path

    def setup_env(self) -> None:
        """
        First time setup for environment variables.

        This method initializes environment variables by prompting the user for their values, iterating over a predefined list of environment items.
        If the user provides no input, a recommended default value is used.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        for (key, prompt_text), recommended_value in self._environment_items:
            value: str = temp if (temp := input(f"Enter {prompt_text} (Enter for {recommended_value}): ")) else recommended_value
            set_key(dotenv_path=self.path, key_to_set=key, value_to_set=value)

    def update_env(self) -> None:
        """
        Update existing environment variables.

        This method allows users to review and edit environment variables specified in a dotenv file.
        It iterates through each variable, displaying its current value and prompting the user to decide if they want to change it.
        If the user chooses to edit a variable, they are prompted to enter a new value.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        env_values: OrderedDict = dotenv_values(self.path)

        for key in env_values:
            is_edit: bool = True if input(f"Value for {key} is {env_values[key]} Want to edit it ? (y/N)") else False

            if is_edit:
                updated_value: str = input(f"Enter updated value for {env_values[key]} : ")
                set_key(self.path, key, updated_value)
