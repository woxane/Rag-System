from openai import OpenAI

class Chatbot:
    def __init__(self, openAI_base_url, openAI_api_key, model_name):
        self.openAi = OpenAI(base_url=openAI_base_url , api_key=openAI_api_key)
        self.model_name = model_name

    def create_prompt(self, context, user_query):
        return f"Using the context provided, answer the question: {user_query}\n\nContext:\n{context}\n\nAnswer:"

    def get_response(self, messages, temperature=0.7, stream=True):
        completion = self.openAi.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = temperature,
            stream=stream
        )

        return completion


    @staticmethod
    def check_chatbot_params(base_url: str, api_key: str, model_name: str) -> bool | str:
        openai = OpenAI(base_url=base_url, api_key=api_key)

        try:
            models = [data.id for data in openai.models.list().data]
            return False
            # if model_name in models:
            #     return False
            #
            # else:
            #     return "Model name provided is not in the list !"

        except Exception as e:
            return str(e)
