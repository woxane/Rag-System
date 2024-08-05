from openai import OpenAI

class Chatbot:
    def __init__(self, openAI_base_url, openAI_api_key, model_name):
        self.openAi = OpenAI(openAI_base_url , openAI_api_key)
        self.model_name = model_name

    def create_prompt(self, context, user_query):
        return f"Using the context provided, answer the question: {user_query}\n\nContext:\n{context}\n\nAnswer:"

    def get_response(self, prompt , temperature = 0.7):
        completion = self.openAi.chat.completions.create(
            model = self.model_name,
            messages = [
                {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
                {"role": "user", "content": prompt}
            ],
            temperature = temperature,
        )
        response = completion.choices[0].message
        return response.content
