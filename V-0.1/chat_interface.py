import streamlit as st

class ChatInterface:
    def __init__(self, document_processor, vectorizer, milvus_handler, chatbot):
        self.document_processor = document_processor
        self.vectorizer = vectorizer
        self.milvus_handler = milvus_handler
        self.chatbot = chatbot

    def display_chat(self, messages):
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def run(self):
        st.set_page_config(page_title="Chat with AI", layout="wide")
        st.title("Interactive Chat with Llama 3.1")
        chat_placeholder = st.empty()

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        st.header("Upload File")
        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:
            chunks = self.document_processor.load_pdf(uploaded_file)
            vectors = self.vectorizer.vectorize(chunks)
            self.milvus_handler.save_vectors(vectors, chunks)

        st.header("Chat with AI")

        self.display_chat(st.session_state.messages)

        if user_input := st.chat_input("Type your message here ..."):
            query_vector = self.vectorizer.vectorize([user_input])
            search_results = self.milvus_handler.search_vectors(query_vector)
            relevant_texts = [res['entity'].get("text") for res in search_results[0]]
            context = "\n\n".join(relevant_texts)
            prompt = self.chatbot.create_prompt(context, user_input)
            st.session_state.messages.append({'role': 'user', 'content': user_input , 'rag_prompt': prompt})

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                messages = [ {"role": message["role"], "content": message["rag_prompt"]
                if message["role"] == "user" else message["content"]} for message in st.session_state.messages ]
                completion = self.chatbot.get_response(messages)

                for response in completion:
                    if response.choices[0].delta.content:
                        full_response += response.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)


            st.session_state.messages.append({'role': 'assistant', 'content': full_response})
