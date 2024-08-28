from typing import List, Dict


def encode_history(user_header_tag: str, assistant_header_tag: str, histories: List[Dict[str, str]]) -> str:
    """
    Encodes a list of user-assistant history exchanges into a formatted string.

    This function takes in a list of conversation histories, Then formats each history into a structured string
    with specified header tags for the user and assistant,
    appending each interaction to a final string that is returned.

    Parameters:
    user_header_tag (str) : A header tag that will be placed above each user's message.
        For Example: <|eot_id|><|start_header_id|>user<|end_header_id|>

    assistant_header_tag (str) : A header tag that will be placed above each assistant's response.
        For Example: <|eot_id|><|start_header_id|>assistant<|end_header_id|>

    histories (List[Dict[str, str]]): A list of dictionaries, where each dictionary represents a single interaction
        between the user and the assistant. Each dictionary must have the keys "user" and "assistant".

    Returns (str): A formatted string that concatenates all user and assistant messages, each prefixed by
        their respective header tags.
    """

    history_structure: str = ""

    for history in histories:
        history_structure += user_header_tag + "\n"
        history_structure += history["user"] + "\n"
        history_structure += assistant_header_tag + "\n"
        history_structure += history["assistant"] + "\n"

    return history_structure
