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
        First time setup environment variable.

        This method set value for settings from user with iterating in _environment_items.

        Parameter:
        None

        Returns:
        None
        """
        for (key, prompt_text), recommended_value in self._environment_items:
            value: str = temp if (temp := input(f"Enter {prompt_text} (Enter for {recommended_value}): ")) else recommended_value
            set_key(dotenv_path=self.path, key_to_set=key, value_to_set=value)

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
