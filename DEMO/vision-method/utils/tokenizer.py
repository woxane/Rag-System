from typing import List, Dict


def encode_history(user_header_tag: str, assistant_header_tag: str, histories: List[Dict[str, str]]) -> str:
    """
    Encode conversation history into a formatted string.

    This function processes a list of exchanges between a user and an assistant, formatting each entry with specified header tags for clarity. It creates a structured string that represents the entire conversation history.

    Parameters:
    -----------
    user_header_tag : str
        A header tag to be placed above each user's message. For example: <|eot_id|><|start_header_id|>user<|end_header_id|>.

    assistant_header_tag : str
        A header tag to be placed above each assistant's response. For example: <|eot_id|><|start_header_id|>assistant<|end_header_id>.

    histories : List[Dict[str, str]]
        A list of dictionaries, where each dictionary represents a single interaction. Each dictionary must contain the keys "role" (indicating either 'user' or 'assistant') and "content" (the message text).

    Returns:
    --------
    str
        A formatted string that concatenates all user and assistant messages, each prefixed by their respective header tags, excluding any responses marked as 'assistant' with references.
    """

    history_structure: str = ""

    for history in histories:
        if history['role'] == 'user':
            history_structure += user_header_tag + "\n"

        elif history['role'] == 'assistant_without_references':
            history_structure += assistant_header_tag + "\n"

        elif history['role'] == 'assistant':
            # do not add the responses with the references and html tags
            continue

        history_structure += history["content"] + "\n"

    return history_structure
