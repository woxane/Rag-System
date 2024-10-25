import streamlit as st
from dotenv import load_dotenv, set_key, dotenv_values
from document_processor import DocumentProcessor
from vectorizer import Vectorizer
from milvus_handler import MilvusHandler
from chatbot import Chatbot


dotenv_path = ".env"
load_dotenv(dotenv_path)

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


    def respond(self, user_input):
        query_vector = self.vectorizer.vectorize([user_input])
        search_results = self.milvus_handler.search_vectors(query_vector)
        relevant_texts = [res['entity'].get("text") for res in search_results[0]]
        context = "\n\n".join(relevant_texts)
        prompt = self.chatbot.create_prompt(context, user_input)
        st.session_state.messages.append({'role': 'user', 'content': user_input, 'rag_prompt': prompt})


    def run(self):
        st.title("PDF Helper")

        # Initialize session state for pdf_texts and messages if not already present
        if "files_id" not in st.session_state:
            st.session_state.files_id= []

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "file_names" not in st.session_state:
            st.session_state.file_names = []

        with st.sidebar:
            st.header("Upload PDF Files")
            current_files = []
            uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
            uploaded_ids = [file.file_id for file in uploaded_files]

            if uploaded_files:
                for id in st.session_state.files_id:
                    if id not in uploaded_ids:
                        #This file has deleted
                        self.milvus_handler.delete_vectors(id)
                        st.session_state.files_id.remove(id)

                for file in uploaded_files:
                    current_files.append(file.name)
                    if file.file_id not in st.session_state.files_id:
                        #New file uploaded
                        chunks = self.document_processor.load_pdf(file)
                        vectors = self.vectorizer.vectorize(chunks)
                        self.milvus_handler.save_vectors(vectors, chunks, file.file_id)
                        st.session_state.files_id.append(file.file_id)
                        st.session_state.file_names.append(file.name)

                        st.success("PDF files uploaded successfully!")
            else:
                #Delete last remaining id
                for id in st.session_state.files_id:
                    self.milvus_handler.delete_vectors(id)

                st.session_state.files_id = []

            st.write("File status:")
            for file_name in st.session_state.file_names:
                if uploaded_files and file_name in current_files:
                    st.write(f"{file_name}")
                else:
                    st.write(f"{file_name} - Removed")

        self.display_chat(st.session_state.messages)

        # Accept user input
        if user_input := st.chat_input("Type your message here ..."):
            self.respond(user_input)

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                messages = [{"role": message["role"], "content": message["rag_prompt"]
                if message["role"] == "user" else message["content"]} for message in st.session_state.messages]
                completion = self.chatbot.get_response(messages)

                for response in completion:
                    if response.choices[0].delta.content:
                        full_response += response.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)


            st.session_state.messages.append({'role': 'assistant', 'content': full_response})


if __name__ == "__main__":
    env_values = dotenv_values(dotenv_path)

    document_processor = DocumentProcessor(chunk_size=int(env_values['chunk_size']),
                                           chunk_overlap=int(env_values['chunk_overlap']))
    vectorizer = Vectorizer(model_name=env_values['embedding_model_name'])
    milvus_handler = MilvusHandler(collection_name=env_values['collection_name'],
                                   dimensions=vectorizer.dimension,
                                   milvus_uri=env_values['milvus_uri'])
    chatbot = Chatbot(openAI_base_url=env_values['openAI_base_url'],
                      openAI_api_key=env_values['openAI_base_url'],
                      model_name=env_values['LLM_model_name'])


    chat_interface = ChatInterface(document_processor, vectorizer, milvus_handler, chatbot)
    chat_interface.run()