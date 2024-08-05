from openai import OpenAI

class Chatbot:
    def __init__(self, openAI_base_url, openAI_api_key, model_name):
        self.openAi = OpenAI(openAI_base_url , openAI_api_key)
        self.model_name = model_name

    def create_prompt(self, context, user_query):
        return f"Using the context provided, answer the question: {user_query}\n\nContext:\n{context}\n\nAnswer:"

    def get_response(self, messages, temperature=0.7, stream=True):
        completion = self.openAi.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = temperature,
            steram=stream
        )

        return completion
